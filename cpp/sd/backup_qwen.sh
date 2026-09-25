#!/bin/bash
# =============================================================================
# backup_qwen.sh — Qwen-Image-2.1 HiRes 两阶段出图（Qwen 原生配方 v2）
# 用法: ./backup_qwen.sh "prompt" [output.png] [width] [height]
#       PRESET=<预设名> ./backup_qwen.sh [output.png] [width] [height]
#       ./backup_qwen.sh --list-presets        # 列出全部风格预设
# 环境变量: PRESET, CFG, STEPS, HIRES_STEPS, HIRES_STRENGTH, HIRES_UPSCALER,
#           SAMPLING_METHOD, SCHEDULER, VAE_TILE_SIZE, VAE_TILE_OVERLAP,
#           OFFLOAD, POSTPROC, CLARITY, SHARPEN, SMART_SHARPEN, EDGE_SHARPEN,
#           FREEU, REALISM, MODEL_DIR
# =============================================================================
#
# 【v2 相对旧版 backup_qwen.sh 的修正】（旧脚本已删除，保留变更说明）
#   1) scheduler: discrete → flux（sd.cpp 对 VERSION_QWEN_IMAGE_2_1 的默认调度器
#      就是 FLUX_SCHEDULER；官方示例不传 --scheduler。强制 discrete 会明显掉画质）
#   2) 脚本层去掉 SD1.5 味的 quality prefix（"masterpiece, best quality, 8k uhd,
#      professional portrait, medium shot"）。注意: img_hires 二进制内置同文前缀
#      默认仍会自动加（NO_QUALITY_PREFIX=1 默认透传 --no-quality-prefix 彻底关闭）;
#      需要时 NO_QUALITY_PREFIX=0 可恢复
#   3) 去掉 negative 里的 SD1.5 textual inversion（embedding:EasyNegative /
#      embedding:bad-hands-5）——Qwen3-VL 文本编码器没有这些 embedding
#   4) 默认关闭重后处理（clarity/sharpen/smart/edge）——避免过锐、塑料感
#   5) steps 提到 25→20，cfg 6.0 / euler（与官方 qwen_image_2.1.md 一致）
#
# 【v3 人像写实固化（2026-09-22 combo B）】
#   - cfg 6.0 / euler / scheduler flux / steps 25→50 / hires strength 0.5
#   - HiRes 放大器 latent-bislerp（更锐），base 2048x1152 → 2560x1440
#   - 后处理默认开启：clarity 0.3 / sharpen 0.3 / smart 0.5 / edge 2.0
#   - 正向自动追加写实词（REALISM=0 关）；负向加 anime/cartoon/illustration/
#     3d render 等反动漫词 + 皮肤油腻词
#   - FreeU 默认关（Qwen 为 DiT，FreeU 空操作；FREEU=1 可开）
#   - POSTPROC=0 关闭后处理；OFFLOAD=0 默认权重上卡（显存紧张时 =1 留 RAM）
#   - NO_QUALITY_PREFIX=1 默认（关 img_hires 内置 masterpiece 前缀, 避免动漫化）
#
# 【v4 甜点定稿（2026-09-24, Qwen 自身 14 组离散扫描, 图 ~/qwen_scan_20260924/）】
#   方法对齐 backup.sh: 固定 seed=25630 / 2560×1440 / 同一人像提示词 /
#   EasyCache / NO_QUALITY_PREFIX=1 / 单因素轮换 14 组。
#   甜点值（默认即此, 裸跑复现）:
#     CFG=6.0  Steps=20→40  HiRes strength=0.4  clarity=0.15  edge-sharpen=0.0
#     upscaler=latent-bislerp  Sampler=euler  Scheduler=flux  POSTPROC=1
#     SEED 默认时间戳随机（复现用 SEED=25630）
#
#   【甜点参数范围（每项单独扫过的离散区间, 括号内为安全值）】
#   cfg           6.0       (5.0~6.5)
#     原理: 引导尺度。z_image 甜点 3.0 不适用 Qwen。实测:
#           4.0 眼神发虚/皮肤略灰; 5.0 自然但眼部对比不足;
#           6.0 眼神锐、肤色正（官方值）; 7.0~8.0 开始 beauty-retouch/CG 感
#           （虹膜过亮、皮肤过匀）。>8 未扫, 预期更假。
#   base steps    20        (18~25)
#   hires steps   40        (35~50)
#     原理: 二次采样细节量。20→30 略糊; 25→50 / 30→50 与 20→40 差距很小
#           （蒸馏 DiT 不吃步数, 与 z_image 结论一致）; 甜点仍 20→40。
#   strength      0.4       (0.35~0.5)
#     原理: HiRes 二次改写幅度。0.3 皮肤纹理偏糊; 0.5 纹理更实;
#           0.6 开始轻微构图漂移。甜点 0.4, 求稳可 0.35, 求纹理可 0.5。
#   clarity       0.15      (0.10~0.20)
#     原理: 局部对比度。0 皮肤死平（与 z_image 同）; 0.3 纹理更跳但
#           略偏「精修」; 0.15 平衡。不要 >0.3（塑料感）。
#   edge-sharpen  0.0       (必须 0)
#     原理: 轮廓高通锐化。白底剪影必出白边/振铃, 与 z_image 同结论。
#   postproc      开         (1; 追求极致自然可试 0)
#     原理: 14_nopp 关后处理更「生」, 开(clarity0.15+柔和 sharpen) 微纹理
#           更立体。重参数(sharp/edge)仍按上表约束。
#
#   【14 组对比要点（seed 25630, Q5, 2026-09-24）】
#     1. 决定性分水岭是 CFG: 4↔6 眼神/肤色差一档, 7+ 开始假面感。
#     2. strength/steps/clarity 都是细调, 不改变构图; CFG 会改五官神态。
#     3. 与 z_image 最大差异: 引导尺度不同（3.0 vs 6.0）, 其余甜点几乎同构
#        （0.4 / 20→40 / clarity0.15 / edge0 / bislerp）。
#     4. FreeU 对 Qwen(DiT) 空操作; 质量前缀会拉回动漫（v2 已默认关）。
#     5. EasyCache 默认开, 不改变甜点参数选择。
#
#   归档: ~/qwen_scan_20260924/{01_base..14_nopp}.png + scan.log
#   复扫: 见 /tmp/opencode/bench/qwen_scan/run_scan.sh（GROUPS 勿用, 用 SCAN_CASES）
#
# 【风格提示词预设库（2026-09-25, 19 组批量验证 + 3 组新增, seed=42）】
#   用法: PRESET=<名字> ./backup_qwen.sh out.png 2560 1440
#   预设自动 REALISM=0（关写实后缀, 风格图不需要；REALISM=1 可覆盖）。
#   适用域: Qwen 擅长风格/插画/带字/复杂构图; 写实人像用 backup.sh (z_image)。
#   实测结论: 19/19 全部可用, 平均 ~4.5 分/张 (2560×1440, EasyCache)。
#   新增 3 组（devana/lighthouse/swordswoman, 2026-09-25, 各横竖 2 张验证:
#   ~/devana_*.png ~/lighthouse_*.png ~/swordswoman_*.png, ~4.5 分/张）。
#
#   ── 机甲科幻 ──
#   mecha            机甲特写: 水墨+3D 混合, 白底, 战损金属, 蓝眼红面颊
#   ── 动漫角色 ──
#   oni_kimono       鬼角和服少女回眸: 蓝白冰晶主题, 华丽发饰纹身
#   miku             初音未来桌前托腮: 兔耳发带, 半闭眼, 慵懒
#   yoru             电锯人 Yoru: 红瞳戒指, 赛博黑白高对比特写
#   morimee          morimee/sinozick 风: 阳光少女动感姿, 鲜艳电影感
#   manga_mono       漫画线稿单色: 圆框眼镜仰视, 极简平涂白底
#   pocahontas       白底简洁人物半身: 黑发长裙项链
#   blue_armor       蓝肤尖耳奇幻战士: 森林平涂, 指向镜头
#   swordswoman      暗黑剑士坐水晶台: 油画质感+动漫赛璐璐, 绿瞳绿甲单色系
#   ── 艺术插画 ──
#   bradhamel        Bradhamel 风超现实侧脸: 金丝曼陀罗, 羊皮纸+靛蓝
#   impression_poppy 印象派罂粟花田: 厚涂油画+水彩, 青蓝金配色
#   folk_art         蜡笔民间艺术: 彩色小镇, 黑裙少女侧影行走
#   blue_wildflower  蓝花田背影: 版画刻线+有限色, 日式书封诗意
#   arabesque        夜景抽象女性: 伊斯兰花纹, 几何+花卉
#   devana           战神 Devana 徽章: 水粉+Scavengers Reign 风, 铜甲圆章斯拉夫花纹
#   ── 立体纸雕 ──
#   papercraft       多层剪纸冬景: 纸纹水彩, 蓝灰金, 物理景深
#   ── 绘本/概念 ──
#   penguin          绘本风企鹅: 夏威夷衫+空白 thought bubble,
#                    梦幻森林洞穴, 水粉平涂+拼贴层叠
#   bustdaal         宝可梦风蝴蝶: 云海浪尖宽景, 无人风景
#   ── 风景/氛围 ──
#   northern_lights  极光冰山: 赛博动漫电影感, 粉彩色
#   rajampat         水墨多色调 Raja Ampat 落日: 高密度 maximalist
#   lighthouse       超现实拼贴灯塔浮岛: 复古编辑拼贴色, 纸纹+北欧极简
#   ── 时尚摄影 ──
#   red_editorial    红唇心形眼线编辑肖像: 硬光红毛皮, 高对比棚拍
#
#   复现: SEED=42 PRESET=<名字> ./backup_qwen.sh ...
#
# 【分辨率】Qwen 要求宽高为 32 的倍数；脚本按 /32 计算 base。
# 【显存】默认 OFFLOAD=0 权重全上卡（2560×1440 实测峰值 ~13.9G/20G, 快 ~3%）;
#   显存不够或多任务时 OFFLOAD=1 权重留 RAM（峰值 ~8.1G, 稍慢）。
# 【采样加速】与 backup.sh 同步默认开 EasyCache（CACHE_MODE=disabled 关）；
#   CUDA graphs 由 build_sd_dl.sh 后端级生效，无需脚本参数。详见 backup.sh 头注释。
# =============================================================================
set -euo pipefail

RED="\033[0;31m"; GREEN="\033[0;32m"; YELLOW="\033[1;33m"
BLUE="\033[0;34m"; CYAN="\033[0;36m"; NC="\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-/data/models/image}"
SD_CLI="${SD_CLI:-$SCRIPT_DIR/build/img_hires}"
SD_BACKEND_DIR="${SD_BACKEND_DIR:-/opt/sd/build-dl/bin}"

DIFFUSION_MODEL="${DIFFUSION_MODEL:-$MODEL_DIR/qwen-image-2.1-Q6_K.gguf}"
LLM_MODEL="${LLM_MODEL:-$MODEL_DIR/Qwen3VL-8B-Instruct-Q4_K_M.gguf}"
VAE_MODEL="${VAE_MODEL:-$MODEL_DIR/qwen_image_2.1_vae_bf16.safetensors}"

# 运行环境依赖内聚到脚本内, 外部无需再 export
export LD_LIBRARY_PATH="$SCRIPT_DIR/build:$SD_BACKEND_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GGML_BACKEND_PATH="${GGML_BACKEND_PATH:-$SD_BACKEND_DIR/libggml-cuda.so}"

ARGS=()
for arg in "$@"; do ARGS+=("$arg"); done

# --list-presets: 打印预设名列表后退出（不跑模型）
if [ "${ARGS[0]:-}" = "--list-presets" ]; then
    echo "风格预设 (PRESET=<name>):"
    echo "  mecha oni_kimono miku yoru morimee manga_mono pocahontas blue_armor swordswoman"
    echo "  bradhamel impression_poppy folk_art blue_wildflower arabesque devana"
    echo "  papercraft penguin bustdaal northern_lights rajampat lighthouse red_editorial"
    exit 0
fi

# 查预设: PRESET=名字 → PROMPT 从 case 表取出; 未设 PRESET 时保持位置参数
preset_prompt() {
    case "$1" in
    # ── 机甲科幻 ──
    mecha)
        echo '(dynamic action pose:1.2), (detailed ink illustration:1.3), white background masterpiece, 3D render, science_fiction, extreme close-up portrait, solo, looking_at_viewer, no_humans, robot, mecha, mobile_suit, v-fin, white and gray armored plating, intricate multi-layered design, internal mechanical components visible through gaps in frame, glowing blue eyes, vibrant red accent on lower face, damaged texture, weathered metallic surface, battle-worn aesthetic, industrial complexity, dramatic lighting, solo, looking_at_viewer, blue_eyes, no_humans, glowing, portrait, robot, glowing_eyes, mecha, close-up, science_fiction, v-fin, mobile_suit'
        ;;
    # ── 动漫角色 ──
    oni_kimono)
        echo 'masterpiece, best quality, amazing quality, very aesthetic, absurdres, highres, extremely detailed, intricate details, year 2025, newest, safe, 1girl, solo, @sw33t, upper body, from behind, from side, looking back, over shoulder, head tilt, (white hair:1.2), blue hair, black hair, gradient hair, medium hair, straight hair, blunt bangs, glossy hair, (oni horns:1.3), horns, (blue eyes:1.3), glowing eyes, sharp eyes, beautiful detailed eyes, long eyelashes, detailed face, blue eyeshadow, teardrop, parted lips, glossy lips, pale skin, very pale skin, (elaborate hair ornament:1.3), kanzashi, hair flower, blue flower, white flower, tassel, dangle earrings, silver jewelry, ornate kimono, black and blue kimono, japanese clothes, floral pattern, back tattoo, ornamental tattoo, bare shoulders, (blue theme:1.3), ice blue, silver accents, (decorative pattern background:1.2), frost pattern, swirling pattern, snowflakes, blue background, high contrast, dramatic lighting, glossy, detailed background, detailed five fingers'
        ;;
    miku)
        echo '@xmxr, masterpiece, best quality, score_7, 1girl, hatsune miku, blue hair, medium hair, twintails, hair between eyes, hair intakes, purple eyes, purple pupils, colored eyelashes, eyelashes, closed mouth, chestnut mouth, furrowed brow, fake animal ears, rabbit ears, white hairband, red bowtie, black ribbon, pink ribbon, hair ribbon, heart hair ornament, x hair ornament, necktie, shirt, tie clip, aqua hair, aqua nails, aqua necktie, long sleeves, half-closed eyes, head on hand, shoulder tattoo, tattoo, upper body, hand resting on table, white undershirt, collared shirt, white shirt, white detached sleeves, lips'
        ;;
    yoru)
        echo 'score_9, score_8_up, score_7_up, Expressiveh, skindentation, detailed face, realistic, masterpiece, best quality, (bokeh:1.2), masterpiece, ultra-HD, very aesthetic, 8K, high detail, highres, BREAK dynamic feel, detailed illustration, detailed background, drop shadow, perfect anatomy, cyberpunk, BREAK (black and white theme), smooth plastic shading, ambient occlusion, sharp focus, beautiful face, perfect face, cute face, adult woman, smooth skin, ((1girl)), ((red_eyes)), ((solo)), ((looking_at_viewer)), ((black_hair)), (jewelry), (glowing), (ring), (glowing_eyes), close-up, yoru_(chainsaw_man), red_theme, multiple_rings, ringed_eyes, own_hands_together, gloves, portrait, blurry, bracelet, long_hair, body_writing, scar_on_face, cross_scar, scar, yor_briar, black_gloves, short_hair, scar_on_cheek, medium_hair, eye_focus, parted_lips, bright_pupils, sidelocks'
        ;;
    morimee)
        echo 'score_9, score_8_up, score_7_up, morimee_style, cute face, skindentation, realistic, masterpiece, best quality, detailed face, intricate details, vivid colors, cinematic lighting, 20yo girl, solo, happy, joyful, dynamic pose, thighs, hips, shorts, braids, bracelets, tank top, navel, necklace, (masterpiece, detailed:1.2) best_quality, ultra-detailed, newest, extremely detailed, sharp focus, best quality, amazing quality, very aesthetic, soft focus, masterpiece, highres, absurdres, sinozick style'
        ;;
    manga_mono)
        echo 'masterpiece, best quality, amazing quality, very aesthetic, absurdres, highres, extremely detailed, intricate details, year 2025, newest, safe, 1girl, solo, @sw33t, upper body, from below, looking at viewer, head tilt, parted lips, long hair, black hair, wavy hair, blunt bangs, (grey eyes:1.2), beautiful detailed eyes, long eyelashes, detailed face, pale skin, blush, (round eyewear:1.3), glasses, thin frame glasses, choker, sleeveless shirt, grey shirt, collared shirt, jacket off shoulders, (manga:1.2), thin lineart, clean lineart, minimal color, muted colors, (monochrome:1.1), soft shading, light blush of color, white background, simple background, detailed, detailed five fingers'
        ;;
    pocahontas)
        echo 'masterpiece, best quality, 1girl, pocahontas, black hair, long hair, black eyes, dark skin, necklace, dress, upper body, solo, looking at viewer, simple background, white background'
        ;;
    blue_armor)
        echo 'masterpiece, best quality, highres, absurdres, newest, SemiFrealism, 748cmstyle, upper body, foreshortening, 1girl, solo, l4thr1lmtg, pointy ears, colored skin, blue skin, brown hair, long hair, purple eyes, facial tattoo, purple sclera, no pupils, armor, gold circlet, cape, green cape, breastplate, fur trim, pauldrons, pointing at viewer, determined, flat colors, outdoors, forest'
        ;;
    swordswoman)
        echo 'The painting style is highly textured, with thick, brush-like strokes and a rough, gritty quality. oil painting (medium), full-body anime character illustration set against a plain white background, featuring a dark swordswoman seated on a pedestal of crystalline structures and flowers. The female character sits elegantly with crossed legs, her body turned slightly toward the viewer. Her face is completely enveloped in pitch-black shadow, except for a single glowing green eye that glares at the viewer. Long, pale seafoam-green hair flows dynamically around her in large, sweeping strands. She wears a gothic black dress adorned with a dark flower pin, patterned sleeve trim, dark tights, and black high heels. Black gloved hands feature sharp, bright green manicured nails. She is seated among white crystalline shards and monochrome lily flowers. Behind her, a massive circular fan-like halo framed with smoke outlines her upper body. Two katanas are positioned around her, one resting near her right hand and another strapped behind her back. Features clean line art, crisp cell shading, and high-contrast monochrome tones with vibrant green eye and nail accents.'
        ;;
    # ── 艺术插画 ──
    bradhamel)
        echo 'Bradhamel art style. A surreal portrait of a woman in profile, her face rendered in soft, aged parchment tones with delicate, flowing lines and ornate golden filigree emerging from her hair and eyes; her right eye is partially open, revealing a deep blue iris with intricate swirling patterns, while the left side of her face dissolves into an elaborate, symmetrical mandala-like structure composed of dark navy, gold, and orange hues, adorned with floral and geometric motifs that cascade downward like dripping ink or liquid metal; the background is a textured blend of sepia and indigo, with faint vertical text columns on the right edge resembling ancient script; the overall composition is vertically oriented, grounded by stylized mountainous forms at the bottom, and illuminated by a mysterious, ambient glow that highlights the metallic gold outlines against the deep shadows, evoking a sense of mystical elegance and arcane beauty.'
        ;;
    impression_poppy)
        echo 'A digital painting in a contemporary impressionism style featuring a woman with her body positioned at an angle amidst a landscape of poppy flowers and ground. The artwork incorporates thick oil paint textures and watercolor-style washes across the scene. The color palette is composed of teal, blue, and golden yellow. A shallow depth of field creates a soft blur on the distant background elements. Layered transparency effects create highlights on the edges of the poppies. Digital texture overlays are visible throughout the composition to simulate an aged, luminous aesthetic'
        ;;
    folk_art)
        echo 'Young woman shown in side profile, with straight, shoulder-length black hair walking across the center of the frame. She wear a simple, long-sleeved black dress and carry a small pink handbag, her eyes closed in a peaceful expression. The setting features a dense hillside town of brightly colored, block-shaped houses in vibrant shades of pink, orange, yellow, and blue under a solid blue sky. In the foreground, the subject walks along a dark path lined with stylized red, white, and pink flowers. The illustration is created in a whimsical folk-art style, characterized by heavy, grainy textures resembling oil pastels or wax crayons on canvas. The lighting is bright and even, emphasizing a saturated color palette and a cheerful, imaginative mood through simplified geometric shapes and bold, flat compositions.'
        ;;
    blue_wildflower)
        echo 'A poetic monochrome-and-blue illustration of a woman seen from behind, standing waist-deep in a field of tall delicate blue wildflowers, her short black bobbed hair blown slightly to one side, pale exposed shoulders and upper back glowing against a soft cream paper background, wearing a simple black sleeveless slip dress with both thin shoulder straps resting neatly in place, the dark fabric blending downward into the stems and shadows. The foreground is filled with fine engraved botanical linework: long grasses, thin stems, daisy-like blue flowers at different heights, intricate black hatching and delicate contour marks, creating a dense meadow texture around her body. Beyond the field, rolling hills stretch horizontally in layered bands of ivory, pale grey, muted blue, and deep navy, with scattered dark tree silhouettes and distant wooded ridges under a quiet speckled sky. The composition is vertical and intimate, the figure placed slightly left of center, facing away into the landscape as if paused in memory, with her black dress and hair forming a strong graphic shape against the pale countryside. Limited palette of cream, black, charcoal, faded indigo, and vivid cornflower blue, combining Japanese book-cover elegance, vintage botanical engraving, soft watercolor washes, hand-drawn ink detail, flat poster-like tonal areas, melancholic rural stillness, wind-touched solitude, fragile romantic atmosphere, quiet visual restraint, refined decorative texture.'
        ;;
    arabesque)
        echo 'abstract, beautiful, strange, woman, night, city, wonderful, arabesque style, geometric patterns, floral, Islamic art, detailed'
        ;;
    devana)
        echo 'detailed gouache painting :: Scavengers Reign digital 2D surreal science-fiction animated still :: Single-subject, luxpunk brutalism, portrait of the goddess Devana, full-body, standing in dynamic contrapposto. A beautiful woman with long dark copper-colored curls cascading over antique bronze armor. In her hand she holds a heavy copper shield engraved with a flat icon of a white-stag'\''s head, horns arcing upward and around the curved edge of the shield in intricate detail. The entire scene is framed inside a perfect circular medallion, painted with cracked plaster texture and mineral pigments. Outside the circle, at the edges and in the corners of the square, dense and ornate floral ornamentation inspired by Slavic pottery -- warm terracotta palette, aged patina, dramatic mythic energy :: limited, restrained, crisp, clear, high-contrast lighting, fine sketch-like luminous strokes, dark empty background emphasizing the vibrant colors, ethereal aesthetic, highly detailed linework, glowing contours, calm expression, minimal dark background, ultra detailed strokes, smooth gradients, colorful luminous textures, dreamy atmosphere, soft glow lighting, sketch-inspired rendering, high contrast, modern digital art :: minimal soft shadow forms and localized edge highlights. The ambient light creates an ethereal glow, while cosmic glow effects amplify the dreamlike atmosphere. Soft, radiant hues bathe the object'\''s features, blending seamlessly with the surrounding world for a mesmerizing, otherworldly composition, chromatic aberration :: no text. image only.'
        ;;
    # ── 立体纸雕 ──
    papercraft)
        echo 'A multi-layered 3D papercraft sculpture of a winter landscape, centered on a large tree. The foreground features a gently sloping hill covered in textured white snow and a winding stream flowing toward the base of the tree. The art style is a fusion of precision-cut paper engineering and digital illustration, characterized by distinct overlapping layers that create physical depth. The color palette uses deep blues and ethereal grays for the snowy terrain and trees, while the background elements feature subtle warm gold undertones. The sky features high-contrast colors between the horizon and the upper atmosphere. Textures include visible paper grain, hand-drawn leaf outlines on individual cut pieces, and watercolor washes across the skys gradient. Lighting is soft and ambient, casting small shadows beneath the layers of paper to emphasize the three-dimensional construction. The scene is captured in a sharp focus that highlights the edges of the cutout elements integrated into a photorealistic environment.'
        ;;
    # ── 绘本/概念 ──
    penguin)
        echo 'A whimsical surreal storybook illustration of a small cute penguin standing on pale rocks inside a dreamlike enchanted forest grotto, wearing black sunglasses pushed up on its head and a bright tropical Hawaiian shirt covered in bold flowers, one flipper raised thoughtfully to its beak with a confused expression, round eyes looking upward at a floating thought bubble. The environment behind the penguin is an ethereal painterly forest with tall pale tree trunks, white bare branches, soft aqua and cream waterfalls cascading down layered mossy stone, clusters of rounded rocks, delicate plants, ghostly ferns, tiny white flowers, and deep black vertical gaps that feel like outer space filled with faint stars. A white deer stands quietly in the upper background between the trees, surrounded by several glowing circular moonlike orbs climbing along a trunk, giving the scene a strange mystical fairytale logic. Palette of misty ivory, bone white, muted sage, dusty teal, pale turquoise, warm ochre, soft beige, and deep cosmic black, with loose watercolor-gouache textures, flat graphic shapes mixed with translucent washes, gentle bloom around pale elements, layered collage-like depth, soft edges, dreamy children-book absurdism, awkward comic timing, serene fantasy atmosphere contrasted with the penguin deadpan confusion.'
        ;;
    bustdaal)
        echo 'BustDaal, bug, butterfly, cloud, waves, no_humans, sky, flying, outdoors, animal_focus, scenery, wide_shot, water, cloudy_sky, ocean, wings, pokemon_(creature)'
        ;;
    # ── 风景/氛围 ──
    northern_lights)
        echo 'Northern lights, ice, icebergs, soft pastell colors, expanse, longing, cinematic, trending on ArtStation, award winning, a masterpiece, ultra quality, 8k, best quality, a masterpiece, award winning, anime, cyberpunk, DB4RZ, DB4RZ style painting, dreadmirthart, in the style of cksc'
        ;;
    rajampat)
        echo 'Ink illustration, multiple tones, aged bright multicolored paper, psychodelic scene, breathtaking beauty of Raja Ampat by sunset, waterfall, shadowed Palms, flying paradise bird, a canoe on the river, glowing, best quality, realistic, whimsical, fantastic, splash art, intricate detailed, hyperdetailed, maximalist style, photorealistic, concept art, sharp focus, harmony, serenity, tranquility, soft pastell colors, ambient occlusion, cozy ambient lighting, surreal, will-o'\''-the-wisp, moonlit, lonely, solitude, windy, tall trees, willows, willowy, OverallDetail, extremely detailed, UHD, long exposure, dystopian but extremely beautiful, best quality, award winning, a masterpiece, Special Ink-drawing mode, animeniji, Mh1$AgThS2, 2D flat anime, cartoon-style, intricate linework with expressive contrasts, soft lighting with dynamic highlights, a masterpiece, award winning, pingtu style, illustration-fen'
        ;;
    lighthouse)
        echo 'A surreal abstract collage featuring a solitary white lighthouse standing on a floating rocky island suspended above a tranquil turquoise ocean. Gentle waterfalls cascade from the island into soft layers of clouds below, creating a peaceful dreamlike atmosphere. A monumental warm orange sun glows behind the lighthouse against a soft blush pink sky, partially veiled by fluffy cream white clouds. Minimalist black geometric stairways rise toward the floating island, adding a striking architectural element. Bold textured rock formations frame the composition while oversized pastel flowers, delicate botanical cutouts, and subtle tropical foliage decorate the foreground like handcrafted paper collage elements. Calm aqua water reflects the pastel sky with soft geometric ripples leading toward the horizon. The entire artwork uses a harmonious retro palette inspired by vintage editorial collage, featuring blush pink, peach, warm coral orange, turquoise, aqua, seafoam green, mint, soft cream, ivory, muted charcoal black, buttery yellow, and subtle lavender accents. Flat geometric forms blend with organic botanical shapes, layered paper textures, vintage grain, clean negative space, Scandinavian minimalism, mid century modern design, contemporary surreal collage art, elegant balanced composition, calming atmosphere, museum quality, a breathtaking masterpiece, award winning'
        ;;
    # ── 时尚摄影 ──
    red_editorial)
        echo 'A close-up editorial portrait of a woman with vibrant ruby red lipstick and graphic red eyeliner detailed with tiny hearts. A hand across her forehead casts a stark shadow, leaving one piercing eye illuminated under high-contrast studio lighting. Her dewy skin is framed by two tight dark braids and a voluminous textured halo of bright red faux fur, creating a dramatic interplay of hard shadows and rich warm colors.'
        ;;
    *)
        return 1
        ;;
    esac
}

# PRESET 模式: 从预设表取 prompt, 默认关写实后缀（REALISM=1 可覆盖）
if [ -n "${PRESET:-}" ]; then
    PROMPT="$(preset_prompt "$PRESET")" || { echo -e "${RED}Error: unknown PRESET '$PRESET' (用 --list-presets 查看)${NC}" >&2; exit 1; }
    REALISM="${REALISM:-0}"
    OUTPUT_FILE="${ARGS[0]:-}"
    WIDTH="${ARGS[1]:-1024}"
    HEIGHT="${ARGS[2]:-1024}"
else
    PROMPT="${ARGS[0]:-solo,single woman,half body portrait of a young woman, soft natural lighting, elegant pose, studio lighting, sharp eyes, pure white background, fair skin, pale skin, smooth skin, matte skin, porcelain skin, flawless skin, medium close up}"
    OUTPUT_FILE="${ARGS[1]:-}"
    WIDTH="${ARGS[2]:-1024}"
    HEIGHT="${ARGS[3]:-1024}"
fi

if [[ "$OUTPUT_FILE" == ~* ]]; then OUTPUT_FILE="${HOME}${OUTPUT_FILE:1}"; fi

die() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }
check_file() { [ -f "$1" ] || die "not found: $1"; }

[ -x "$SD_CLI" ] || die "img_hires not found: $SD_CLI"
check_file "$DIFFUSION_MODEL"
check_file "$LLM_MODEL"
check_file "$VAE_MODEL"

[[ "$WIDTH" =~ ^[0-9]+$ ]] && [ "$WIDTH" -gt 0 ] || die "width must be positive integer"
[[ "$HEIGHT" =~ ^[0-9]+$ ]] && [ "$HEIGHT" -gt 0 ] || die "height must be positive integer"

echo -e "${GREEN}✓ All checks passed${NC}"

CFG_SCALE="${CFG:-6.0}"
STEPS="${STEPS:-20}"
HIRES_STEPS="${HIRES_STEPS:-40}"
HIRES_STRENGTH="${HIRES_STRENGTH:-0.4}"
SAMPLING_METHOD="${SAMPLING_METHOD:-euler}"
SCHEDULER="${SCHEDULER:-flux}"
HIRES_UPSCALER="${HIRES_UPSCALER:-latent-bislerp}"
VAE_TILE_SIZE="${VAE_TILE_SIZE:-32}"
VAE_TILE_OVERLAP="${VAE_TILE_OVERLAP:-0.5}"
OFFLOAD="${OFFLOAD:-0}"
POSTPROC="${POSTPROC:-1}"
CLARITY="${CLARITY:-0.15}"
SHARPEN="${SHARPEN:-0.3}"
SMART_SHARPEN="${SMART_SHARPEN:-0.5}"
EDGE_SHARPEN="${EDGE_SHARPEN:-0.0}"
FREEU="${FREEU:-0}"
REALISM="${REALISM:-1}"
# 默认关质量前缀（masterpiece/best quality 会把 Qwen 拉向动漫）；NO_QUALITY_PREFIX=0 可开
NO_QUALITY_PREFIX="${NO_QUALITY_PREFIX:-1}"
# 采样步缓存（与 backup.sh 同步）: 默认开; CACHE_MODE=disabled 关
CACHE_MODE="${CACHE_MODE:-easycache}"
CACHE_THRESHOLD="${CACHE_THRESHOLD:-0.2}"
CACHE_START="${CACHE_START:-0.15}"
CACHE_END="${CACHE_END:-0.95}"
REALISM_SUFFIX="photorealistic, realistic photograph, raw photo, natural skin texture"

# v2：不再自动加 booru quality prefix（需要时可用 QUALITY_PREFIX 显式指定）
if [ -n "${QUALITY_PREFIX:-}" ] && [[ "$PROMPT" != *"masterpiece"* ]]; then
    PROMPT="$QUALITY_PREFIX, $PROMPT"
fi

# v3：写实约束（Qwen 默认偏动漫，追加写实关键词；REALISM=0 关闭）
if [ "$REALISM" = "1" ] && [[ "$PROMPT" != *"photorealistic"* ]]; then
    PROMPT="$PROMPT, $REALISM_SUFFIX"
fi

NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-blurry, low quality, worst quality, jpeg artifacts, noise, bad anatomy, deformed, watermark, text, logo, signature, oily skin, shiny skin, greasy skin, glossy skin, plastic skin, skin blemishes, anime, cartoon, illustration, painting, drawing, 3d render, cgi, anime face, cel shading}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -n "$OUTPUT_FILE" ]; then
    if [[ "$OUTPUT_FILE" == *"/"* ]]; then
        OUTPUT_DIR=$(dirname "$OUTPUT_FILE"); BASE=$(basename "$OUTPUT_FILE")
    else
        OUTPUT_DIR="$HOME"; BASE="$OUTPUT_FILE"
    fi
    OUTPUT="${BASE%.png}_${TIMESTAMP}.png"
else
    OUTPUT_DIR="$HOME"
    MD5=$(echo "$PROMPT" | md5sum | cut -c1-8)
    OUTPUT="${TIMESTAMP}_${MD5}.png"
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_PATH="$(cd "$OUTPUT_DIR" && pwd)/$OUTPUT"

round32() { echo $(( ( ($1) / 32 ) * 32 )); }
if [ "$WIDTH" -eq 3840 ] && [ "$HEIGHT" -eq 2160 ]; then
    LOW_W=2560; LOW_H=1440
elif [ "$WIDTH" -eq 2560 ] && [ "$HEIGHT" -eq 1440 ]; then
    LOW_W=2048; LOW_H=1152
elif [ "$WIDTH" -eq 1920 ] && [ "$HEIGHT" -eq 1080 ]; then
    LOW_W=1536; LOW_H=864
elif [ "$WIDTH" -eq 1280 ] && [ "$HEIGHT" -eq 720 ]; then
    LOW_W=1024; LOW_H=576
else
    LOW_W=$(round32 $(( WIDTH * 4 / 5 )))
    LOW_H=$(round32 $(( HEIGHT * 4 / 5 )))
    [ "$LOW_W" -lt 512 ] && LOW_W=512
    [ "$LOW_H" -lt 512 ] && LOW_H=512
fi

echo ""
echo "========================================"
echo "  Qwen-Image-2.1 HiRes Fix (native v2)"
echo "========================================"
echo -e "Target Size: ${GREEN}${WIDTH}x${HEIGHT}${NC}"
echo -e "Low-res Pass: ${GREEN}${LOW_W}x${LOW_H} -> ${WIDTH}x${HEIGHT}${NC}"
echo -e "Steps: $STEPS -> $HIRES_STEPS (HiRes)"
echo -e "CFG Scale: ${CYAN}$CFG_SCALE${NC}"
echo -e "HiRes Strength: $HIRES_STRENGTH"
echo -e "HiRes Upscaler: ${CYAN}$HIRES_UPSCALER${NC}"
echo -e "Sampler: ${CYAN}$SAMPLING_METHOD${NC} + ${CYAN}$SCHEDULER${NC}"
echo -e "VAE Tiling: ${VAE_TILE_SIZE} overlap ${VAE_TILE_OVERLAP}"
echo -e "Post-processing: ${POSTPROC} (0=off), realism=${REALISM}"
echo -e "FreeU: ${FREEU} (DiT 空操作)"
echo -e "Offload to CPU: ${OFFLOAD}"
if [ "$CACHE_MODE" != "disabled" ]; then
    echo -e "Cache: ${CYAN}$CACHE_MODE${NC} threshold=$CACHE_THRESHOLD range=[$CACHE_START,$CACHE_END]"
fi
echo "----------------------------------------"
echo -e "Prompt: ${YELLOW}$PROMPT${NC}"
echo -e "Output: ${GREEN}$OUTPUT_PATH${NC}"
echo "========================================"
echo ""

SEED="${SEED:-$(date +%s)}"
echo "Generating...  $(date '+%H:%M:%S')"

SD_CMD=("$SD_CLI"
  --diffusion-model "$DIFFUSION_MODEL"
  --llm "$LLM_MODEL"
  --vae "$VAE_MODEL"
  --negative "$NEGATIVE_PROMPT"
  --cfg "$CFG_SCALE"
  --method "$SAMPLING_METHOD"
  --scheduler "$SCHEDULER"
  --diffusion-fa
  --vae-tiling
  --vae-tile-size "$VAE_TILE_SIZE"
  --vae-tile-overlap "$VAE_TILE_OVERLAP"
  -W "$LOW_W" -H "$LOW_H"
  --steps "$STEPS"
  --hires
  --hires-width "$WIDTH"
  --hires-height "$HEIGHT"
  --hires-strength "$HIRES_STRENGTH"
  --hires-steps "$HIRES_STEPS"
  --hires-upscaler "$HIRES_UPSCALER"
  --cache-mode "$CACHE_MODE"
  --cache-threshold "$CACHE_THRESHOLD"
  --cache-start "$CACHE_START"
  --cache-end "$CACHE_END"
  -s "$SEED"
)

if [ "$POSTPROC" -eq 1 ]; then
    SD_CMD+=(--clarity "$CLARITY" --sharpen "$SHARPEN" --sharpen-radius 1
             --smart-sharpen "$SMART_SHARPEN" --smart-sharpen-radius 2
             --edge-sharpen "$EDGE_SHARPEN" --edge-sharpen-radius 2
             --edge-sharpen-threshold 0.3)
else
    SD_CMD+=(--clarity 0 --sharpen 0 --smart-sharpen 0 --edge-sharpen 0)
fi
if [ "$FREEU" -eq 1 ]; then
    SD_CMD+=(--freeu --freeu-b1 1.3 --freeu-b2 1.4)
fi
if [ "$OFFLOAD" -eq 1 ]; then
  SD_CMD+=(--offload-to-cpu)
fi
if [ "$NO_QUALITY_PREFIX" -eq 1 ]; then
  SD_CMD+=(--no-quality-prefix)
fi

SD_CMD+=("$PROMPT" "$OUTPUT_PATH")

START_TIME=$(date +%s)
( cd "$SD_BACKEND_DIR" && "${SD_CMD[@]}" )
END_TIME=$(date +%s)
GEN_DURATION=$((END_TIME - START_TIME))

fmt_duration() { local s=$1; [ $s -ge 60 ] && echo "$((s/60))m $((s%60))s" || echo "${s}s"; }

if [ -f "$OUTPUT_PATH" ]; then
    echo ""
    echo "========================================"
    echo -e "${GREEN}✓ Generation successful!${NC}"
    echo -e "File:   ${GREEN}$OUTPUT_PATH${NC}"
    echo -e "Size:   ${BLUE}$(du -h "$OUTPUT_PATH" | cut -f1)${NC}"
    echo -e "Time:   ${YELLOW}$(fmt_duration $GEN_DURATION)${NC}"
    echo -e "Seed:   ${YELLOW}$SEED${NC}"
    echo "========================================"
else
    echo ""
    echo -e "${RED}✗ Generation failed! Output file not found${NC}"
    exit 1
fi
