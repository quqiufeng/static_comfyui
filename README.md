# ComfyCLI — StaticPy 前端 + stable-diffusion.cpp 后端的 ComfyUI 编译重写

用 **StaticPy** 1:1 复刻 ComfyUI 的节点编排与 DAG 调度，底层推理全部交给 **stable-diffusion.cpp**（GGML/CUDA），最终编译成**独立 ELF 二进制**，零 Python 运行时依赖。

> **给 AI 阅读者：如何理解 StaticPy 语言**
>
> StaticPy 是面向 AI Agent 的 ML 编译语言，语法是 Python 子集 + FFI 扩展。
> `staticpy/` 下的编译器核心是上游 `/opt/ReScheme` 的**原样拷贝**，项目侧只维护胶水。
>
> 阅读以下文件即可掌握这门语言：
>
> 1. **翻译器** — [`staticpy/static_translate.py`](./staticpy/static_translate.py)
>    - `typecheck_module()` 定义支持/不支持的 Python AST 节点（类型检查）
>    - `translate_function()` / `translate_block()` 定义 Python 语法到 Scheme 的翻译规则
>    - `BUILTIN_MODULES` 列出内置模块映射；`extern fn` 提取逻辑在文件开头的正则匹配中
> 2. **运行时** — [`staticpy/static_prelude.scm`](./staticpy/static_prelude.scm)
>    - 值类型映射：int→fixnum、float→flonum、bool→boolean
>    - 数组/dict/文件 I/O/JSON/字符串工具的实现
> 3. **标准库 FFI** — [`staticpy/static_stdlib.scm`](./staticpy/static_stdlib.scm)
>    - `foreign-procedure` 如何映射 `extern fn`
> 4. **项目胶水** — [`comfycli/comfycli_ffi.scm`](./comfycli/comfycli_ffi.scm)
>    - `load-shared-object` 加载 `libsdcpp_adapter.so` + 上游缺失的内置
>
> **本地对翻译器的 3 处补丁**（`is None` / `break-continue` / 模块全局）见 [BUILD.md](./BUILD.md)。
> **学习路径：**翻译器定义"能写什么"→ 运行时定义"编译成什么"→ 胶水定义"能调什么"。

## 远景

ComfyUI 是优秀的 Stable Diffusion 工作流引擎，但 Python 解释器带来可移植性痛点：

| 问题 | 影响 |
|------|------|
| `pip install` 依赖地狱 | 多项目共用 venv 冲突，复现难 |
| Python 运行时开销 | 节点调度、prompt 解析的解释器开销 |
| 部署体积大 | 需要完整 Python 环境 + torch + 数十个 pip 包 |
| 打包困难 | PyInstaller/Nuitka 打包 torch 动辄 2GB+，且兼容性差 |

**ComfyCLI 方案：静态编译编排层，复用成熟 C++ 推理后端。**

- 用 **StaticPy** 重写 ComfyUI 的编排逻辑（节点、DAG、调度、链接解析）
- 用 **stable-diffusion.cpp** 承担所有推理计算（UNet、VAE、CLIP、采样、ControlNet、LoRA 等）
- 通过 **FFI** 调用 `libsdcpp_adapter.so` 的 C API
- 编译产物：**独立 ELF 二进制**（+ `libsdcpp_adapter.so` + GLIBC 兼容层），零 Python
- 用 **code search（my_db）** 语义索引辅助定位源码，加速 1:1 翻译

## 提示词库（backup_qwen.sh 出图用）

实战收集的 42 条风格提示词，配套 `cpp/sd/backup_qwen.sh` 使用（Qwen-Image-2.1，2560×1440 甜点配方）：

```bash
REALISM=0 SEED=42 bash cpp/sd/backup_qwen.sh '<提示词>' ./output/out.png 2560 1440
```

> 说明：原稿中的 `<lora:...>` 标签为 ComfyUI/Kohya 专用语法，sd.cpp 的 Qwen 管线**不支持**，已全部移除；需要 LoRA 效果请改用风格描述词。负面词建议去掉 `painting/illustration/drawing` 等与画风冲突的词（见 `backup_qwen.sh` 默认负面词）。

| # | 中文说明 | 提示词 |
|---|---|---|
| 1 | **月下仙子与蝴蝶光粒**<br>月夜森林，侧坐巨型红蘑菇上的仙子吹散蝶群化作光粒；水彩+油画梦幻质感，适合 2560×1440 壁纸 | `A delicate young fairy sits gracefully on the broad cap of an enormous red woodland mushroom in an enchanted moonlit forest. She is shown in elegant side profile with one long bare leg hanging freely and her head slightly raised. Her softly long flowing brown hair is blowing in the wind, decorated with tiny blossoms. She wears a short translucent dress covered in fine shimmering threads and botanical details. Two enormous butterfly wings extend behind her, intricately patterned like stained glass in dusty rose, peach, lavender, periwinkle blue and subtle antique gold. She raises one hand toward her lips and gently blows a swirling cloud of colorful butterflies into the air. Pink, violet, blue, orange and golden butterflies dissolve into luminous sparkling particles. A huge creamy ivory full moon glows behind her. The misty forest fades into muted blue gray, sage and teal shadows with delicate wildflowers below. Dreamlike fantasy watercolor and oil painting, luminous glazing, soft bleeding pigments, delicate impasto accents, fine botanical detail, subtle canvas texture, ethereal diffuse light, romantic painterly realism, pastel jewel tones, magical atmospheric haze, elegant storybook fine art, cinematic composition, ultra detailed, award winning, a breathtaking masterpiece` |
| 2 | **栈桥日落剪影**<br>厚涂油画马赛克色块：少女深色剪影坐湖边木栈桥，落日与几何倒影，ArsMJ/Mosaic 风 | `Vibrant abstract impasto oil painting of a solitary young woman sitting quietly on the edge of a rustic wooden pier beside a calm lake at sunset. She appears only as an elegant dark silhouette, seen from behind, wearing a long flowing dress. Her very long hair streams freely behind her in the gentle evening wind. She gazes across the water toward a low glowing golden sun on the horizon. The lake mirrors the intense sunset colors in broken geometric reflections. Dark graceful trees frame the left side, dense flowering vegetation surrounds the shoreline, small birds cross the sky. Brilliant mosaic like blocks of crimson red, vermilion, burnt orange, golden yellow, turquoise, cobalt blue, emerald green, coral pink and violet. Expressive oil painting, thick impasto, palette knife texture, layered rectangular color patches, visible brushwork, dripping pigments, luminous reflections, decorative flowers, dreamy contemporary abstraction, poetic, peaceful, radiant, canvas texture, ultra detailed, award winning, a breathtaking masterpiece, ArsMJStyle, MosaicArsMJStyle, Mosaic` |
| 3 | **意大利卡普里岛俯瞰**<br>60 年代度假感俯拍：红白条纹伞、泳池、绿松石海湾与帆船，maximalist splash art | `Create an image of the pitoresque and famous village Capri in Italy, 60`s,, swimming pools, red and white striped umbrellas, view from above, ships off the coast, surrounded by turquoise clear water. The art style is highly detailed and painterly, take care of the right dimensions, emphasizing the interplay of light, water, and texture to evoke a sense of serenity and timeless beauty, best quality, realistic, captivating, intricately detailed, hyper detailed, maximalist style, fantastic, splash art, intricate detailed, concept art, bright colors, clear turquoise water, colorful, superb composition, sharp focus, high contrast, stylized, clear, colorful, masterpiece award winning, CAICO` |
| 4 | **荷塘独木舟顶视**<br>垂直俯视：少女抱书侧卧独木舟，巨幅睡莲叶包围，中心青绿水面透出水下石纹 | `Dreamlike top down fantasy scene viewed from a perfectly vertical overhead camera, looking straight down onto a narrow weathered wooden canoe drifting through a vast deep lotus pond. The entire canoe is visible and positioned near the center, surrounded by immense round lotus leaves in many sizes. Inside, a young woman sleeps peacefully on her back along the length of the boat. Her head is turned gently to one side, long dark wavy hair spread across the wooden boards. One arm bends above her head while the other rests across her torso holding a small closed book. She wears a flowing ivory dress with soft layered fabric cascading around her legs, her bare feet visible near the stern. Pale pink lotus blossoms and buds emerge between dense emerald leaves. A luminous turquoise opening in the vegetation surrounds the canoe, revealing clear water, submerged rocks and delicate caustic light patterns below. The pond becomes progressively darker toward the edges, shifting into deep teal, petrol and midnight blue. Refined fantasy illustration, smooth painterly digital brushwork, rich botanical detail, softly modeled forms, subtle luminous highlights, atmospheric depth, elegant color transitions, tranquil magical light, serene and mysterious, award winning, a breathtaking masterpiece` |
| 5 | **万圣木乃伊装天使**<br>Sargent 油画+Rembrandt 光影：白翅金发少女提糖果袋，神秘明暗氛围（已去 lora） | `GrimmsPaint, SplashPastoDaal, DRKM4GE, Masterpiece, A teen girl angel with a cute shy smile, large feathery wings, long blond hair, gold glowing halo, an illustration with an upper body-to-upper angle, a oil paint style, and a cute pose, she is wearing a mummy Halloween costume, holding a bag of Halloween candy, The painting shows shadow and light, Masterpiece, intricate lines, intriguing atmosphere, sharp magnificent details, delicate features, elaborate details, ultra detailed, romantic, light and shadow, painted by John Singer Sargent, Masterpiece, Rembrandt lighting, brush stroke, lovely` |
| 6 | **武士拔刀侧影**<br>wabisabi 枯笔狂放线条：日之丸背景、侧面收刀瞬间、飘发逆风、战斗架势 | `wabisabi, japan, a woman, samurai, holding a katana, sheathing, from side. In the background is a sun resembling the Japanese flag, with slightly faded, pale colors. wind, stray hair. A ferociously energetic illustration style built on wild, gestural linework that captures raw movement every stroke, loose ink-driven cross-hatching layered with confident painterly color, chaotic yet controlled compositions bursting with kinetic tension, richly textured surface detail giving textures a visceral, almost sculptural intensity, and a bold, untamed painterly ferocity that feels equally at home in fine art and manga tradition. Stand up straight and stick out her chest in fighting stance.` |
| 7 | **和服少女嗅白花**<br>wabisabi 氛围、体积光过曝：侧颜嗅白花、彩绘背景、士郎正宗 Shirow 风 | `wabisabi, japan, (ethereal:2), volumetric lighting, backlit, over exposure. A photo-realistic shoot from a profile camera angle about a young woman in a traditional japanese kimono smelling a flower with paint like background mixture. That mixture creates an atmosphere of mystery and subtlety. the woman has Crystal-clear white skin. the image also shows a soft, blurred white background with some flowers and delicate patterns. on the middle of the image, a young woman with fair skin and dark hair styled in an elegant updo, wearing a traditional japanese kimono with a floral pattern in shades of white, yellow, and orange. she appears to be smelling a white flower, with her eyes closed and a peaceful expression on her face. she is holding a small white flower in her right hand. her hair is adorned with a small, delicate hair comb and a pink ribbon, adding a touch of elegance to her overall look. her kimono is intricately designed with a floral pattern, and she is wearing a brown sash around her waist, which adds to her traditional japanese attire. shirow, Shirow Masamune style` |
| 8 | **灰发青瞳男性胸像**<br>pinterestcore04 蚀刻线描+棕褐淡彩：M 形长刘海、高领黑大衣银肩甲、锐利直视 | `pinterestcore04. The rendering style utilizes fine, consistent line weight throughout, suggesting an ink drawing or etching quality, with minimal shading achieved through hatching, stippling and varying line density to define contours and volume. limited palette; sepia colors rendered in faded watercolors. upper body bust shot. subject is a male figure with very long grey hair with curtained bangs in a tall 'M' shape that then sweep down to frame his face. he has bright turqoise-blue eyes, staring intensely at the viewer. he wears a high-collared black coat that is open in the front, revealing two crossed straps over a bare chest. over the black coat he wears angular silver pauldrons. the jacket is cinched at the waist. grey patterned inset border.` |
| 9 | **雪落浮世绘艺妓**<br>kastusika hokusai / 浮世绘：背面视角、颈后特写、华丽发簪、流动笔触 | `@kastusika hokusai, shunga, flowing blush stroke turning into ukiyo-e, snow falling, oiran hair ornament, japanese traditional hairstyle, A shot from behind, fair skin, the nape of the neck` |
| 10 | **民间艺术灯塔海报**<br>木刻+丝网印：几何巨浪环绕孤灯塔、同心光束，赤陶/芥末/深青有限色块 | `Contemporary folk art graphic poster, linocut and screen-print aesthetic, an isolated lighthouse standing on a narrow rocky island, enormous stylized waves surrounding it as repeating geometric patterns, birds reduced to simple graphic symbols, circular beam of light radiating outward in concentric ornamental shapes, no realistic ocean rendering, terracotta red, mustard yellow, deep teal, ochre, black and warm cream, strong geometric composition, limited color palette, bold carved contours, layered flat shapes, subtle halftone and ink grain, imperfect handmade registration, textured paper background, poetic solitude, sophisticated European folk poster design` |
| 11 | **哈比少女水墨降落**<br>白发红瞳耳机哈比、展翅降落羽毛飘落；sumi-e 飞白水墨与宣纸质感 | `A harpy girl character design, tomboyish look, wearing large over-ear headphones, very long feathery red eyelashes, white short hair with red highlights at the tips, wearing a loose white hoodie, smile red eye, looking at viewer, beauty mark under eyes. Around her neck hangs a traditional necklace featuring a red thread weaving through the square holes of several small antique bronze coins, the coins textured with age and patina, sleeveless. flying big wings arms spreading mid air as she is landing with her feather falling. abstract composition, serene mood, intricate linework, subtle color palette, vivid Zen Ink Painting, sumi-e style with bold black ink washes against textured rice paper, dynamic negative space, minimalist yet detailed feather rendering` |
| 12 | **少女与白天鹅浅浮雕**<br>象牙白单色 chalk/浮雕：少女与天鹅相依，发丝羽毛织物化作卷曲装饰波纹 | `Create a chalk painting image with a soft, ethereal quality that features of a serene beautiful young woman kneeling closely beside an enormous majestic white swan, their bodies forming an intimate flowing composition. The swan gently curves its long elegant neck toward her, its head resting near her face while its enormous wings spread partially around her like a protective embrace. The woman has delicate features and extremely long wavy hair flowing freely behind her. Her eyes are softly closed and her head inclines toward the swan with peaceful tenderness. She wears an elegant flowing gown whose layered fabric gradually merges with the swan feathers. Hair, feathers and fabric transform into sweeping ornamental waves that curl around both figures. The entire scene is sculpted as an exquisite monochrome ivory bas relief. Pearl white, warm alabaster and subtle champagne shadows. Intricate carved feathers, flowing hair strands, delicate fabric folds, smooth porcelain surfaces, dimensional sculptural depth, soft directional illumination, romantic classical composition, ethereal tranquility, handcrafted museum quality fine art, whispered, unseen secrets, fantasy, dreamlike, surrealism, sharp focus, rich in detail, dreamy, cinematic, trending on ArtStation, captivating, fantastical, splash art, harmony, superb composition, a breathtaking masterpiece, award-winning, DB4RZ, DB4RZ style painting, in the style of cksc, Papercut` |
| 13 | **月下飞蛾化发少女**<br>侧脸闭眼月光银边，黑发逐渐碎裂成数百只飞蛾；深青风暴云、诗意暗黑 | `Dark cinematic fantasy portrait of a mysterious young woman shown in elegant side profile, her head gently lowered and eyes closed with a serene melancholic expression. Her pale face is illuminated by a huge luminous full moon directly behind her, creating a soft silver rim light along her nose, lips and jaw. Her long black hair forms a vast flowing mass that gradually dissolves into hundreds of dark moths. The transformation begins naturally within individual strands, becoming increasingly fragmented until the entire right side disperses into flying moths at different scales and depths. Some remain sharply detailed near her hair while distant moths fade into the misty sky. Her bare shoulder also dissolves subtly into smoky pigment and tiny wing fragments. Deep teal storm clouds surround the moon with atmospheric fog and subtle watercolor blooms. Midnight blue, petrol, charcoal, muted turquoise and silver. Painterly realism blended with refined digital illustration, delicate facial rendering, expressive textured brushwork, translucent glazing, soft pigment diffusion, cinematic moonlight, dramatic negative space, mysterious and poetic, atmospheric depth, enchanting, mysterious, romantic, a breathtaking masterpiece, award winning` |
| 14 | **黑虎金纹岩上坐**<br>Bradhamel 风：黑虎金色纹理侧坐岩上，雾灰背景、定向光凸显轮廓 | `Bradhamel art style. A majestic black tiger with shimmering golden stripes sits in profile atop a dark, textured rock, gazing forward with intense focus against a soft, misty gray background; the tiger's fur is rendered with fine detail and subtle sheen, its tail curled elegantly over the rock's edge, surrounded by faint silhouettes of grass blades at the base, in a realistic digital illustration style with moody, directional lighting that highlights the gold stripes and contours of the animal's form.` |
| 15 | **彩虹光与樱花瓣少女**<br>NIJISIS 干净赛璐璐动漫：暗背景、彩虹光条与白樱瓣掠过肩发、金光锁骨 | `@NIJISIS, Anime-style illustration shot from a close three-quarter angle, a girl with sleek dark hair and a calm, intense gaze looking directly at the viewer, one shoulder bare beneath a thin strap top. Streaks of vivid rainbow-colored light and scattered white cherry blossom petals burst dynamically across her hair and shoulder, as if caught mid-motion. The composition is set against a deep dark background, with warm golden light glowing softly against her collarbone. The artistic style is a clean, glossy digital anime illustration with crisp linework, smooth cel-shading, and richly saturated jewel-toned colors, evoking a striking, dreamlike, high-energy atmosphere.` |
| 16 | **黑墨蚀刻动态全身**<br>score_9 黑墨蚀刻：全身侧影、手臂向右上大幅伸展、交叉排线明暗对照 | `score_9, score_8_up, score_7_up, tot_art, Global: Detailed black ink etching style, high-contrast monochrome palette with deep tonal washes, dramatic chiaroscuro lighting setup. Medium: Full body composition captured from a sharp side profile, atmospheric void-black space with fine grain texture. Soft, diffused light source creating subtle subsurface scattering on the subject's form. Subject: A solitary female figure executing an intensely dynamic pose, head tilted sharply upward conveying intense focus. Arms are dramatically extended toward the upper right corner in a sweeping, elongated gesture. Lower body depicted with intricate, flowing ink lines suggesting voluminous fabric motion. Detail: Surface texture features fine, layered ink washes and delicate cross-hatching to define form. Focus on the interplay between deep black voids and subtle tonal gradations.` |
| 17 | **夜景套房金发短发**<br>BUN3 数字插画：回眸撩发、挑逗微笑、黑色镂空连体衣、落地窗城市夜灯 | `Character - adult blonde with a tousled pixie cut, smoky eyes and a black cutout bodysuit. Pose - turning over her shoulder, one hand lifting her hair, with a teasing grin. Background - a luxurious hotel suite at night, city lights glowing through the windows. @BUN3, digital illustration` |
| 18 | **蝴蝶耳饰蓝裙少女**<br>NIJISIS：淡蓝抹胸礼服刺绣、蝴蝶耳坠与宝石颈饰、浅蓝条纹背景光尘 | `@NIJISIS, Anime-style illustration of a girl with short pale blonde hair, a warm gentle smile, delicate butterfly-shaped earrings and a jeweled butterfly choker at her neck. She wears an off-the-shoulder pale blue ballgown with intricate floral embroidery, small blue butterflies fluttering around her hair and shoulders. The composition is set against a soft striped light blue background, sparkling light particles drifting through the air. The artistic style is a clean, glossy digital anime illustration with crisp linework, smooth cel-shading, and richly saturated soft blue and cream tones, evoking a delicate, romantic, fairytale-like atmosphere.` |
| 19 | **头顶交通锥少女**<br>pseudomcht 波普动漫 / rakugakingu：扁平亮色、头顶交通锥的戏谑构图 | `pseudomcht, In the pseudomcht art style, rakugakingu, pop anime style, pop art, colorful, 1girl, wearing a traffic cone on her head` |
| 20 | **雾塘黑天鹅群**<br>童话油画：黑天鹅领航白鹅群，睡莲金枝、秋林橙红晨光与梦幻散景光束 | `Whimsical painting of a cute black swan gracefully gliding on a misty pond surrounded by a cluster of pristine white swans, all floating serenely among floating lily pads and glowing goldenrod reeds. The swan's sleek black feathers shimmer with dew-kissed highlights under the soft glow of golden morning light filtering through a canopy of autumn trees whose leaves blaze in fiery orange, crimson, and amber hues. Behind them, a dense forest rises in layers of mossy bark and fallen foliage, blurred into dreamy bokeh with ethereal light beams piercing through the mist.` |
| 21 | **狼女与圆猫工业废墟**<br>m0nm0n 扁平色+细线描：狼女 Gumdong 与圆球猫 Poyopoyo 置于巨构齿轮废墟、荧光菌菇 | `m0nm0n, flat colors with thin, elaborate linework. Gumdong—an anthropomorphic wolf woman with striking crimson-red eyes glowing with fierce intelligence, positioned prominently in the foreground. She has sleek dark fur rendered in deep charcoal-grey with thin linework defining muscular anatomy throughout her form. Her most distinctive feature is a thick, brutal spiked collar encircling her neck, the spikes rendered as sharp angular projections with thin precise linework defining their menacing geometry. She wears a form-fitting black tank top rendered in flat black color with subtle thin fabric texture linework, and ripped, torn denim jeans with ragged edges rendered through irregular thin linework creating a worn, battle-hardened aesthetic. Her powerful posture suggests she is navigating cautiously through dangerous territory. Beside her, dwarfed by her presence, the small Poyopoyo—a round, rotund yellow cat with light yellow fur rendered in soft, even coloring with delicate thin linework throughout suggesting plush texture. His body is perfectly spherical and bulbous, rendered as simple rounded forms with thin contour lines defining his basic shape. His legs are tiny stubby appendages rendered as minimal short lines barely visible beneath his massive round body. His small triangular ears point upward, rendered with thin inner-ear linework. His large, round eyes express innocent curiosity despite the dangerous setting. His mouth is a simple curved line suggesting a gentle, unaware expression. The environment is a massive industrial graveyard—the ruins of a gargantuan manufacturing facility rendered in an impossible scale. Towering metal framework rises dramatically around both characters, constructed from geometric beams and girders rendered in cold steel-grey with thin linework defining the angular, mechanical architecture. Pipes of various diameters extend at impossible angles throughout the space—some vertical, some horizontal, some spiraling upward in helical patterns, all rendered with thin linework showing cylindrical form and metallic surface texture. The machinery is partially intact but massively deteriorated. Giant gears—some the size of buildings—are frozen mid-rotation, their teeth rendered as precise sharp points defined by thin linework in circular arrangements. Conveyor systems hang from above like organic vines, their metal belts rendered with thin cross-hatching linework suggesting woven metal surfaces. Rusted panel sections hang from corroded bolts, their deterioration suggested through irregular thin linework creating rust patterns across their surfaces. The ground is a chaotic landscape of industrial debris—scattered metal plates, broken machinery components, twisted cables, and crystallized corrosion formations rendered throughout. The overall perspective is disorienting, suggesting the scale is so vast that navigation is treacherous. Shadows are rendered in dramatic thin linework, following the complex geometry of the mechanical structures, creating a maze of dark angular shapes across the ground. The atmosphere is oppressive and desolate. Bioluminescent fungal growths emerge from cracks and crevices, rendered in pale sickly greens and blues with thin linework suggesting organic forms colonizing the dead machinery. These create small points of eerie light throughout the industrial tomb, with their glow rendered through selective thin highlight linework. Emotions: Gumdong projects protective determination and cautious wariness; Poyopoyo appears innocent and unaware of the danger. Pattern vocabulary: mechanical, geometric precision, industrial decay, deterioration, angular architecture, mandala-like gear formations.` |
| 22 | **盔甲小猫剑士**<br>Bradhamel 素描风：幼猫持巨剑立于岩台，铅笔排线+炭笔明暗，史诗又可爱 | `Bradhamel art style, Sketch style. A tiny fluffy kitten in miniature armor, holding a sword twice its size in both paws, standing in a heroic stance on a rocky outcrop. Wind whips its fur, a tattered cape flutters behind. Its expression is fiercely determined, eyes narrowed. Rough pencil lines, cross-hatching, ink and charcoal, dynamic gesture strokes, unfinished edges, dramatic chiaroscuro, deep shadows and metallic accents, epic yet adorable mood` |
| 23 | **秋日玫瑰园天使**<br>GrimmsPaint/SplashPasto 油画：金发白翅天使着秋装立于玫瑰园，时尚模特姿态（已去 lora） | `GrimmsPaint, SplashPastoDaal, Masterpiece, high quality, Oil Painting, Hyper-realistic, a pretty blonde teen angel with large white feathery wings wearing a stylish modern fall outfit standing in a rose garden in fall, she looks like a fashion model, light and shadow, painted by John Singer Sargent, Masterpiece, Rembrandt lighting, brush stroke, lovely` |
| 24 | **青叶奇幻大树**<br>zidiusArt 厚涂油画：青白叶大树对撞紫夜空与橙金星云，冷暖高对比 | `zidiusArt, A surreal fantasy landscape painted in a thick impasto oil painting style. A large, majestic tree with a dark, twisting trunk and a canopy of pale cyan, mint green, and white leaves dominates the right side. The background features a striking contrast: the upper left is a deep starry night sky in dark purple, while the lower section glows with an intense, fiery orange and gold light, resembling a nebula or magical fire. The foreground consists of rocky terrain in cool blue and teal tones. Sparkling particles float in the air. Dreamlike atmosphere, high contrast between warm and cool colors, masterpiece, textured brushwork.` |
| 25 | **涂鸦风格女性面孔**<br>混合媒介街头艺术：红镜片大墨镜、蓝纹头巾、泼漆背景、洋红唇，反叛张力 | `A vibrant, expressive mixed-media portrait of a woman's face, rendered in an abstract, graffiti-inspired style. The composition is dominated by the subject's face from the nose up, with her head tilted slightly to her right, creating a dynamic and engaging focal point. Her appearance is defined by bold, stylized features: large, round sunglasses with thick black frames that obscure her eyes, which are depicted as dark, reflective pools of green within the red lenses. Her nose is painted in a simple, orange-brown hue, and her lips are full, parted slightly, and colored in a striking magenta-pink, contrasting sharply against the surrounding palette. The subject's hair is represented by chaotic, energetic strokes of black, white, and gray, suggesting movement or wind, with some strands appearing to be caught in motion. These lines intersect with abstract elements like splatters of paint and faint graffiti symbols, including a peace sign visible near the bottom. Her headwear consists of a heavily textured, patterned bandana or cap painted in swirling shades of blue, teal, and white, adding to the chaotic yet intentional visual style. The artist's signature is subtly placed. The background is an energetic mix of splattered paint—predominantly blues, oranges, whites, and grays—with drips and smears that suggest a raw, unfiltered process. This chaos contrasts with the more defined facial features, drawing attention to them while still immersing the viewer in the overall texture. Lighting appears to be artificial and stylized, with no natural light source visible. The color palette is bold and high-contrast: deep reds, electric blues, stark whites, and vibrant oranges create a visually arresting effect that conveys a sense of urgency or rebellion. The composition is centered on the subject's face, but the chaotic elements around her frame it in a way that feels both contained and overwhelming. The overall mood evokes a sense of confidence, mystery, and urban street culture. The abstract style and vivid colors suggest a fusion of personal identity with artistic expression, possibly hinting at themes of freedom, defiance, or self-expression through art. There is an element of danger or allure in the subject's gaze, as though she is both observing and challenging the viewer simultaneously. The image is clearly a piece of contemporary street art, designed to be visually arresting and emotionally resonant, rather than realistic or documentary in nature. The combination of bold colors, abstract forms, and chaotic textures creates an atmosphere that is both intimate and universal—inviting viewers into a world where identity and emotion are rendered through the language of paint and expression. The painting is signed by the artist, indicating its origin as a deliberate artistic statement rather than a mere depiction. The overall effect is one of powerful visual energy and emotional intensity, designed to provoke thought and reaction from the viewer.` |
| 26 | **白帆帆船破浪**<br>Bradhamel 厚涂油画：深蓝海面白帆竖幅构图，笔触捕捉海面动势 | `Bradhamel art style. A tall sailing yacht with a single large white sail billowing in the wind, cutting through deep blue ocean waves that sparkle with sunlight, under a vast cerulean sky streaked with soft, voluminous white clouds; rendered in thick impasto oil painting style with visible brushstrokes, capturing the dynamic motion of the sea and the crispness of the air, framed vertically to emphasize the vessel's upward trajectory against the horizon, with the boat positioned slightly off-center to create visual tension and depth, the hull reflecting the sky's light as it moves forward through the water.` |
| 27 | **赤肤机械拳魔女**<br>@96yottea 日本画风：深红皮肤黑电路纹、发光弯角、蒸汽朋克拳套与烟斗 | `@96yottea, Nihonga-style painting. A striking female figure with deep red skin, adorned with intricate black circuitry-like line markings that trace across her face and neck. She possesses a pair of long, glowing red, curved horns that curve from her head before rising straight up, glowing with an inner fiery heat. Her expression is a subtle, closed-mouth smile, revealing a single sharp fang. She wears a tight cropped jacket and loose, baggy pants. One normal hand rests casually over her knee, gripping a long smoking pipe. The other hand is encased in a massive, intricate steampunk mechanical gauntlet, complete with pipes and valves bellowing thick white steam. A demon tail with several rotating rings extends behind her. Swirling, very long gradient hair with fiery highlights disintegrates into the air. Cinematic lighting highlights the metallic textures of the gauntlet and the glow of the horns. High contrast, detailed digital illustration style, rich colors, sharp focus, high fidelity.` |
| 28 | **深蓝牡丹对半少女**<br>NIJISIS 特写：一半深蓝牡丹暗部、一半高光白溢，明暗对割构图 | `@NIJISIS, Anime-style close-up illustration shot straight-on, a girl with pale hair and striking blue eyes peeking through her bangs, lips slightly parted, dense deep blue peonies clustered along one side of the frame. The composition splits the image in two: one side dark and richly saturated with the blue flowers and shadowed hair, the other side washed in bright glowing white light bleeding into her pale hair and face, creating a strong contrast between the deep blue floral darkness and the luminous bright side. The artistic style is a clean, glossy digital anime illustration with crisp linework, smooth cel-shading, and richly saturated colors, evoking a serene, otherworldly atmosphere.` |
| 29 | **北欧女武神 Polaris Hilda**<br>银发紫眸、金饰黑甲红裙蓝披、持雷杖；Tsutomu Nihei 线条与纵深 | `Polaris Hilda, a beautiful, tall young woman with long, light gray or silver hair, violet eyes, white skin, wearing rose or lavender lipstick. She wears a Viking-style breastplate in black with gold motifs, gold bracelets, a winged headband also with gold motifs, a long red skirt, and a dark blue cape. She possesses a very long metal spear that serves as her scepter and ceremonial staff. In battle, she uses this spear to launch lightning bolts, energy blasts, or spheres of energy. She rules a frigid environment in the far north of Europe, heavily inspired by the Scandinavian atmosphere of Norse mythology. Focus on the girl, shadow and light, Masterpiece, intricate lines, intriguing atmosphere, sharp magnificent details, delicate features, elaborate details, (2\3 rule composition:0.5), ultra detailed, romantic, lighting, brush stroke, lovely, del1cate_balance style, in the style of Tsutomu Nihei. vantablack, depth of field, high resolution, intricate details, 4k, wallpaper` |
| 30 | **秋日森林河湾剪影**<br>Bradhamel 童书插画风：苔木河湾、暖赭柔光，剪影人物群显自然辽阔 | `Bradhamel art style. The image depicts a serene, autumnal forest scene with a winding river on the right, dense trees with mossy trunks and sparse orange leaves overhead, and a group of silhouetted figures sitting near a fallen log or natural archway in the center-right. The lighting is soft and diffused, casting gentle highlights through the canopy onto the mossy ground and water, while shadows deepen in the forest interior. The overall mood is tranquil and slightly mysterious, evoking a sense of quiet exploration. The style is reminiscent of classic illustrated children's book art, characterized by bold outlines, textured washes of muted greens, browns, and ochres, and a hand-drawn quality that emphasizes depth and atmospheric perspective. The composition frames the figures as small within the vastness of the woodland, emphasizing scale and solitude, with the river's surface reflecting dappled light and pebbles visible along its edge. No additional characters or animals are present beyond the silhouetted group, preserving the original focus on the forest environment and the quiet presence of the figures.` |
| 31 | **通天阶梯与天宫女神**<br>中国神话奇幻：粉云阶梯通向悬浮天宫、女神剪影，珊瑚橙+薄荷绿珠光质感 | `An enormous staircase climbs through pink clouds toward a palace floating directly above the sunrise. A tiny human ascends while a gigantic goddess appears behind the palace like a living celestial silhouette. Use coral-orange and pale mint green, with glossy clouds, translucent silk-like light, pearlescent architecture, glowing gold edges, and radiant sunrise reflections. Heavenly, hopeful yet lonely, monumental scale, Chinese fantasy mythology, polished glazed finish, shell-like shine, 8K, 4K, Ultra HD, 1:1, no text.` |
| 32 | **斯拉夫女战士与狼**<br>中世纪细线描：链甲毛披风持匕首的辫发女战士与狼，雪夜白桦林（已去 lora） | `medieval-style drawing, delicate medieval-style illustration, young Slavic warrior woman with long braided blonde hair adorned with a headband, wearing chainmail and a spotted fur cloak over her shoulders, dagger in one hand, standing beside a wolf with its tongue out in a snowy moonlit forest with bare birch trees, muted earthy color palette, intricate linework, atmospheric night lighting` |
| 33 | **曼陀罗披风持弯刀少女**<br>m0nm0n：明亮赭绿平背景、橙白棕放射曼陀罗披风、猫形唇线、弯刀斜持 | `m0nm0n. Brightly lit composition. A full-length tall, long-legged female figure stands centrally in the composition, facing mostly forward with a slight tilt to the head, presenting a somber or neutral facial expression. moroboshi kirari, upper lips curved in a cat-like, rounded 'w' shape. she has long, voluminous wavy brown hair, with long wavy sidelocks framing her face and bangs covering her forehead. She wears an elaborate garment featuring a black clothes under an elaborately patterned cape; the black underclothes feature high neckline and long sleeves, patterned pants and high-heeled shoes. The cape is of a dark fabric adorned with a radiating circular mandala-like pattern in shades of orange, off-white, and brown. The figure's left arm is held slightly away from the body, grasping a curved blade that extends diagonally downward across her body. The background is a plain flat bright verdigris green.` |
| 34 | **海崖金发动漫电影帧**<br>高预算动画电影感：铂金发少女立草崖边望碧海积云，天蓝/青绿/云白清透配色 | `Anime movie film still, cinematic anime, high-budget animation. Full body shot of a young woman with long platinum-blonde hair streaming dramatically in the ocean wind standing at the edge of a grassy seaside cliff, gazing out at a vast turquoise ocean under a towering sky of stacked cumulus clouds painted in elaborate detail. She wears a simple light-blue linen dress that the wind presses against her figure. Bright, clean natural daylight from a high afternoon sun illuminates the scene evenly, with the grass and her hair catching vivid highlights. The ocean below shimmers with thousands of tiny white light points. Peaceful, free, expansive feeling. Sky blue, turquoise, cloud white, grass green, platinum hair glow. Panoramic composition.` |
| 35 | **峡湾装饰版画极光**<br>ta86_inkriot 细黑线：Art Nouveau 木刻峡湾，金线冰裂、齿轮状同心极光环 | `ta86_inkriot, thin black outlines, Stylized Arctic fjord in decorative woodcut / Art Nouveau style. Foreground of dark-blue ice floes and a frozen inlet with gold linear highlights like engraved ripples. A thin gold path of ice cracks leads toward jagged black cliff silhouettes. Low amber sun on the horizon. The entire sky is a giant concentric aurora-machine: gold and turquoise rings, gears of light, spiral clouds outlined in fine gold. Stars like inlaid gems. Palette: deep indigo, cold turquoise, warm gold and amber. Flat decorative layers, no realistic shading — only line and color.` |
| 36 | **奔跑者与星系巨瞳**<br>电影感暗角：山脊剪影跑者对望祖母绿+金橙光点的宏观银河/巨眼奇观 | `A man jogs along a shadowed ridge toward a colossal, luminous expanse dominating the horizon. The sweeping background resembles a glowing galaxy or macroscopic eye, shimmering with dense, iridescent speckles of emerald, gold, and fiery orange. Cinematic deep black edges create a heavy vignette, contrasting sharply with the monumental scale and radiant pointillist textures of the surreal phenomenon.` |
| 37 | **林中歪屋石桥故事书**<br>apieckstyle 欧洲旧故事书：歪斜林中小屋与石桥、雁群旅人、旧纸质感（已去 lora） | `apieckstyle, a richly detailed vintage European storybook illustration of a crooked woodland cottage beside an old stone bridge, surrounded by bare twisted trees, fallen leaves, wandering geese and a traveler approaching along the muddy path, with delicate uneven ink linework and aged paper texture` |
| 38 | **风暴日落野花田**<br>zidiusArt 厚涂：紫苑花海与琥珀牧草，风暴紫云对撞金橙落日，诗意张力 | `zidiusArt, Create a dramatic oil painting of a wildflower field at sunset, with a stormy sky and strong temperature contrasts. In the foreground, depict a dense cluster of cosmos flowers with large, delicate petals in shades of lilac, violet, and deep magenta, accented by bright yellow centers. Render the petals with soft curves and subtle asymmetry to convey natural beauty, and add fine details like thin stems and delicate foliage. Surround the flowers with tall, dry grasses and wild grains in warm golden and amber tones; use layered, translucent strokes to suggest the play of wind and light. Add subtle highlights on the grass tips to enhance the sense of a low, glowing sun. In the middle ground, paint a vast, gently rolling field fading into the horizon; keep the details soft and atmospheric, with warm ochre and honey tones dominating the earthy palette. For the sky, compose a tumultuous, stormy scene with thick, swirling clouds rendered in bold, impasto brushstrokes. Use a dramatic color mix: deep purples, violets, and dark grays contrasted with fiery oranges, reds, and yellows where the sunset light breaks through. Let the clouds appear heavy and dynamic, with visible texture and movement. Place the sun low on the horizon, partially hidden behind the clouds; let its warm glow illuminate the field and flowers, creating a luminous effect and long, soft shadows. Add subtle lens flare and radiant halos to emphasize the intensity of the light. Balance the composition with a clear horizon line that divides the calm earth from the turbulent sky, and use atmospheric perspective to fade the distant field into a soft haze. Prioritize a tactile, painterly quality: visible brushstrokes, impasto texture in the clouds and flowers, and delicate glazes for the grass and horizon. The mood is dramatic yet poetic, evoking the tension between impending storm and the quiet beauty of a sunset field.` |
| 39 | **低角度神祇拼贴**<br>低角度数字插画：合十而立的无脸神祇、黄蓝纹样服饰，几何撕纸拼贴+复古棕褐高反差 | `masterpiece, best quality, exciplit, "Digital illustration from a low-angle perspective of a stylized deity with mystical aura, standing contemplatively in the middle of the frame, surrounded by abstract shapes and lines in shades of green and yellow. A man, late 40s-early 50s, stands with clasped hands, long spiky black hair in a bun, intense gaze focused distant. He wears intricate patterned yellow and blue outfit, body adorned in blues, greens, yellows, purples. Faceless, wearing a wrist bracelet. High-contrast monochrome and warm sepia tones with hard vintage-photo contrast, layered paper textures, geometric dissection composition, nostalgic, avant-garde mood."` |
| 40 | **动漫海底珊瑚秘境**<br>广角俯看海床：五彩珊瑚与热带鱼群、双海龟掠过、彩虹焦散光斑洒落 | `Anime style, vibrant and dynamic underwater nature scene with a wide-angle, dynamic perspective looking down into the ocean floor, teeming with corals of all colors and intricate shapes—pink, turquoise, gold, and magenta—swaying gently in currents; schools of colorful tropical fish darting through the coral formations, each uniquely patterned and vividly hued; translucent bubbles rising toward the surface; soft, glowing rainbow caustics filtering down from above through the water’s surface, casting shimmering prismatic patterns on the seabed; two sea turtles gliding gracefully overhead, their shells detailed and textured, moving slowly across the upper midground, creating gentle ripples; the entire scene bathed in a dreamy, diffused underwater light that enhances the surreal beauty, with rich saturation and fluid motion suggesting a lively, immersive marine ecosystem.` |
| 41 | **撕纸分层凝视少女**<br>撕纸/叠层动画效果：短刘海少女隔缝凝视、蓝橙异色瞳、无框眼镜、大理石美甲、暖橙主调 | `@748cmstyle, @NJSW33T, (masterpiece, best quality, very aesthetic, amazing quality, absurdres, newest, girl), The image is an animation-style illustration that stands out with a unique effect, as if the paper has been torn or layered. Figure and composition: A girl with short bangs is gazing intensely forward. By utilizing the white space that cuts across the screen, it creates a mysterious feeling as if the girl is gazing out through a gap. Eyes and Glasses: The large, transparent pupils are a subtle blend of sky blue and orange, with intricate highlights that sparkle in the light. She wears clear rimless glasses, and the red blush visible through the lenses adds a soft touch. Nail art and hands: a girl's fingers are placed close together at the bottom of the screen. Each fingernail is adorned with a dazzling marble nail art featuring an abstract blend of orange, blue, and white, adding a modern and stylish fashion element. Color and shading: Warm orange and yellow tones dominate the overall mood, while the blue hues of the eyes and nails serve as refreshing accents. Overall, it is a high-quality illustration that harmonizes a soft airbrush effect with clean line work.` |
| 42 | **海滩日落印花裙少女**<br>score_7/8 动漫上色：晚霞海滩风吹印花长裙、手撩发、肩部镂空（已去 lora） | `absurdres, masterpiece, score_7, score_8, anime coloring, anime screencap, 1girl, print dress, shoulder cutout, long dress, beach, wind, sunset, curvy, hand in own hair, floral print dress` |

## 出图尺寸与 HiResFix 高清重绘要求

### 原理：两阶段 latent 重绘，不是简单放大

HiResFix ≠ 插值放大。流程（`cpp/sd/backup.sh` → `img_hires` → `libsdcpp_adapter.so` → sd.cpp）：

1. **base 阶段**：在较低分辨率（如 1920×1080）跑完整采样，出构图与主体（`steps=20`）。
2. **latent 放大**：直接在 latent 空间插值放大到目标尺寸（`latent-bislerp` 保细节 / `latent-bicubic` 偏软），**不经过像素域**，无放大算法的涂抹感；等价 ComfyUI 的 `LatentUpscale(bislerp)`。
3. **二次采样（denoise 重绘）**：加噪后以 `hires_strength=0.4` 的去噪强度二次采样。sd.cpp 按 sd-webui 语义先把调度总步数放大为 `hires_steps / strength`（40/0.4=100），再截取尾部**恰好 40 步**有效去噪（`request.cpp:449`），既补细节又不破坏构图。等价 `KSampler(denoise=0.4)`。
4. **后处理**：clarity 局部对比 / sharpen / smart_sharpen / edge_sharpen（`sdcpp_adapter.cpp` postproc）。

因此画质取决于三条硬约束：

| 约束 | 要求 | 违反后果 |
|---|---|---|
| **放大倍数** `target/base` | **≤1.33× 最优，1.5× 上限**；base 尽量贴近模型原生 ~1024 | 倍数越大，二次采样要"凭空补"的细节越多，构图漂移/噪点 |
| **base 最小边** | ≥512（脚本自动按比例抬） | base 太小构图不稳 |
| **宽高整除** | 目标与 base 都需被 `vae_scale × model_down_factor` 整除 | 否则 sd.cpp `request.cpp:196` **向上取整**（LOG_WARN），比例微变 |

### 尺寸怎么选：宽高取 64 的倍数

整除要求按模型不同（`diffusion_engine.cpp:2836`）：

- **z_image**（DiT，VAE 8 × down 1）→ **8 的倍数**
- **Qwen-Image-2.1**（VAE 16 × down 2）→ **32 的倍数**
- SD1.5/SDXL（unet 8 × VAE 8）→ **64 的倍数**（本项目主路线不用）

取 **64 的倍数**可一次性满足全部三种，且**除 2 缩档后仍合规**（64k→32k→16k 链条里 32 整除保持得最久）；只跑 z_image + Qwen 则 **32 的倍数**即够。

**推荐尺寸**（base 由 `backup.sh:214` 硬编码表 / `img_hires.cpp:98` 同款公式自动推算，放大均为 ~1.23-1.33×）：

| 用途 | 横版 | 竖版 | base → target |
|---|---|---|---|
| 主力档 | **2048×1152** | **1152×2048** | 1664×960 → 1.23× |
| 主力档（脚本硬编码表） | **2560×1440** | **1440×2560** | 1920×1080 → 1.33× |
| 1080p 替代（严格 64） | 1920×1088 | 1088×1920 | 1536×896 → 1.25× |
| 快速档 | 1280×768 | 768×1280 | 1024×640 → 1.25× |
| 试图 | 640×360 | 360×640 | 512×320 → 1.25× |
| 方图 | 1024×1024 | — | 896×896 → 1.16× |
| 4K（倍数上限） | 3840×2160 | 2160×3840 | 2560×1440 → 1.5× |

注意：`2560×1440 / 1440×2560` 中 1440 是 **32 整除而非 64 整除**——z_image/Qwen 下完全合规并命中硬编码表（最优 1.33×），仅 SDXL 会被抬到 1472。表外自定义尺寸走通用公式（目标 latent ×4/5、对齐到 8 的倍数），只要倍数 ≤1.5× 即可用。

对照关闭缓存与参数扫描区间见下节与 `cpp/sd/backup.sh` 头注释。

## 采样加速（HiRes 出图）

E1xMIN 2560×1440（RTX 3080）从 **~11 分钟 → 3.5 分钟（约 3.2×）**，默认已开：

1. **EasyCache**（主因）：相邻采样步变化小于阈值时复用 latent、跳过本步 DiT forward；蒸馏 turbo 后期步更易命中（base 跳 9/20，hires 跳 ~30/41）。`backup.sh` / `backup_qwen.sh` / `backup_scene.sh` 环境变量 `CACHE_MODE`（默认 `easycache`）、`CACHE_THRESHOLD`（默认 0.2，越低跳越多）。
2. **GGML_CUDA_GRAPHS=ON**：`build_sd_dl.sh` 开启，把一步采样的 CUDA kernel 录成 graph 一次提交，减少 launch 开销。
3. `img_hires` 分段计时：`Model loaded` / `generate wall` / `Post-processing` / `TOTAL wall`，日志含 `EasyCache skipped N/M steps`。

对照关闭缓存：`CACHE_MODE=disabled ./backup.sh ...`。详见 `cpp/sd/backup.sh` 头注释与 `TODO.md`「采样加速」。

## 快速开始

```bash
# 1. 编译
./build.sh

# 2. 运行（workflow 模式，动态后端）
LD_LIBRARY_PATH=cpp/sd/build:/opt/sd/build-dl/bin \
  GGML_BACKEND_PATH=/opt/sd/build-dl/bin/libggml-cuda.so \
  ./comfycli-bin workflow.json --output-dir ./output

# 3. 或 prompt 模式（CLIP 默认取自 checkpoint；如需外置：--clip-l / --clip-g）
LD_LIBRARY_PATH=cpp/sd/build:/opt/sd/build-dl/bin \
  GGML_BACKEND_PATH=/opt/sd/build-dl/bin/libggml-cuda.so \
  ./comfycli-bin --checkpoint /data/models/image/sd_xl_base_1.0.safetensors \
  --prompt "a photo of a cat" --output ./out.png

# 4. 打包部署包（含依赖 .so + GLIBC 兼容层）
GLIBC_TARGET=2.35 ./deploy.sh

# 5. 发送到远程服务器
GLIBC_TARGET=2.35 ./deploy.sh --scp user@remote_host

# 6. 远程运行（零 Python）
ssh user@remote_host "bash /opt/comfycli/run.sh workflow.json --output-dir ./output"
```

### workflow JSON 示例

`test_remote_2560.json` 是一个完整的 HiResFix 工作流，包含 DiffusionModelLoader → CLIPTextEncode → HiResFix → VAEDecode → SaveImage 节点链，支持 FreeU、SAG、VAE Tiling 及后处理（clarity/sharpen/smart_sharpen/edge_sharpen）参数：

```json
{
  "1": {
    "class_type": "DiffusionModelLoader",
    "inputs": {
      "diffusion_model_name": "z_image_turbo-Q5_K_M.gguf",
      "llm_name": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
      "vae_name": "ae.safetensors"
    }
  },
  "2": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "solo,single woman,half body portrait of a young woman, soft natural lighting, elegant pose, studio lighting, sharp eyes",
      "clip": ["1", 0]
    }
  },
  "3": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "blurry, low quality, worst quality, bad anatomy, deformed, watermark, text, logo",
      "clip": ["1", 0]
    }
  },
  "4": {
    "class_type": "HiResFix",
    "inputs": {
      "model": ["1", 0],
      "positive": ["2", 0],
      "negative": ["3", 0],
      "width": 2560, "height": 1440,
      "steps": 20, "cfg": 2.5,
      "sampler_name": "euler", "scheduler": "discrete", "seed": 42,
      "hires_steps": 45, "hires_strength": 0.35,
      "freeu": 1, "freeu_b1": 1.3, "freeu_b2": 1.4,
      "clarity": 0.2, "sharpen": 0.3, "sharpen_radius": 1,
      "smart_sharpen": 0.5, "smart_sharpen_radius": 2,
      "edge_sharpen": 1.5, "edge_sharpen_radius": 2, "edge_sharpen_threshold": 0.3
    }
  },
  "5": { "class_type": "VAEDecode", "inputs": { "samples": ["4", 0], "vae": ["1", 2] } },
  "6": { "class_type": "SaveImage", "inputs": { "images": ["5", 0], "filename_prefix": "output" } }
}
```

使用该 JSON 生成图片：
```bash
LD_LIBRARY_PATH=cpp/sd/build:/opt/sd/build-dl/bin \
  GGML_BACKEND_PATH=/opt/sd/build-dl/bin/libggml-cuda.so \
  ./comfycli-bin test_remote_2560.json --output-dir ./output
```

## 已实现节点

`comfycli/nodes.static.py` 注册 **132 个节点**，覆盖 ComfyUI 全部 **120 个内置节点名（100%）**。

> **对齐度说明（重要）**：节点**名称**已 100% 对齐（132 注册 / 120 内置名）。**行为：115 真实实现 / 17 透传**。
>
> **推理后端原则**：出图一律走 sd.cpp/ggml（专为推理优化：量化、算子融合、低显存）。torch helper 仅用于 ggml 没有的**权重级**操作（合并/导出），不在主出图路径上。
>
> **真实实现**（主要来源）：
> - sd.cpp 原生能力：模型加载、采样、VAE、LoRA、ControlNet、IPAdapter、图像算子、`LoadLatent/SaveLatent`、`PreviewAny`、`Reroute`
> - **sd.cpp patch**：`CLIPSetLastLayer`(clip_skip)、`ModelSamplingFlux/SD3/AuraFlow`(flow_shift)、`ModelComputeDtype`(wtype)、`ModelAttentionBackend`(flash_attn)、`ModelSamplingContinuousEDM/ContinuousV`(sigma 区间)、`RescaleCFG`、`VideoLinear/TriangleCFGGuidance`、`ModelNoiseScale`、`ModelSamplingDiscrete`、区域条件 `ConditioningSetArea/SetAreaPercentage/SetAreaStrength/Multiply`（采样循环逐区合成，显存 ~2.7GB）、`LatentRotate/Flip/Composite/Blend`、`RepeatLatentBatch/LatentFromBatch/SetLatentNoiseMask`
> - **libtorch helper（权重级）**：`ModelMerge*`(20)、`CLIPMerge*`(3)、`CheckpointSave/VAESave/CLIPSave/ModelSave/ImageOnlyCheckpointSave`（可选库 `libcomfycli_torch.so`，缺失时相关节点不可用、其余功能不受影响）
>
> **仍透传/占位（17）**：`ConditioningSetMask`/`SetTimestepRange`、模型专属 conditioning（`AnimaLLLiteApply`/`QwenImageDiffsynthControlnet`/`ZImageFunControlnet`/`WanUni3CControlnetApply`/`SUPIRApply`/`USOStyleReference`/`ConditioningSetAreaPercentageVideo`）、`CLIPVisionEncode`、`StyleModel*`(2)、`unCLIPConditioning`、`GLIGEN*`(2，模块/注入已实现，加载器待完成)、`ModelSamplingStableCascade`、`ModelPatchLoader`、`SVD_img2vid_Conditioning`、`WebcamCapture`。待验证清单见 `TODO.md`。
>
> **部分映射**：`ModelAttentionBackend` 仅 `flash_attn` 有语义；`ModelSamplingContinuousEDM/ContinuousV` 仅对使用 sigma 区间的调度器（`exponential`/`karras` 等）生效，`discrete` 用模型内置 sigma 表、天然忽略区间。
>
> 即"能跑通的工作流范围"取决于 sd.cpp 的能力边界；核心出图链路（txt2img / img2img / inpainting / ControlNet / HiRes / LoRA / IPAdapter）是真实可用的。

### 模型加载

| 节点 | 输出 | 说明 |
|------|------|------|
| `CheckpointLoaderSimple` / `CheckpointLoader` | `MODEL`, `CLIP`, `VAE` | 加载 SDXL/SD1.5 等 checkpoint（safetensors） |
| `UNETLoader` | `MODEL` | 加载单个 diffusion/unet 文件 |
| `VAELoader` / `CLIPLoader` | `VAE` / `CLIP` | 分离组件（透传，sd.cpp 单 context） |
| `DiffusionModelLoader` | `MODEL`, `CLIP`, `VAE` | GGUF diffusion 模型 + LLM 文本编码器（Flux/Z-Image） |
| `DualCLIPLoader` | `CLIP` | 兼容占位（pipeline 内含 CLIP） |

### 条件 / 文本编码

| 节点 | 输出 | 说明 |
|------|------|------|
| `CLIPTextEncode` | `CONDITIONING` | 文本编码（后端 CLIP 内部处理） |
| `CLIPSetLastLayer` | `CLIP` | 真实实现（映射 sd.cpp `clip_skip`） |
| `ConditioningCombine` / `ConditioningConcat` | `CONDITIONING` | 文本拼接 |
| `ConditioningAverage` | `CONDITIONING` | 按强度决定拼接顺序 |
| `ConditioningZeroOut` | `CONDITIONING` | 空条件 |
| `ConditioningSetArea` / `SetAreaPercentage` / `SetAreaStrength` / `Multiply` | `CONDITIONING` | 真实实现（区域条件，torch 管线按区合成） |
| `ConditioningSetMask` / `SetTimestepRange` | `CONDITIONING` | 透传（掩码/时间步分段未实现） |
| `ControlNetApply` | `CONDITIONING` | 应用 ControlNet（控制图 + 强度） |

### 图像 / 潜空间

| 节点 | 输出 | 说明 |
|------|------|------|
| `EmptyLatentImage` | `LATENT` | 空白潜空间 |
| `EmptyImage` | `IMAGE` | 纯色图（OpenCV 生成） |
| `LatentUpscale` / `LatentCrop` | `LATENT` | 修改潜空间尺寸 |
| `LatentRotate` / `LatentFlip` / `LatentComposite` / `LatentBlend` | `LATENT` | 真实实现（OpenCV 图像级旋转/翻转/合成/混合） |
| `RepeatLatentBatch` / `LatentFromBatch` / `SetLatentNoiseMask` | `LATENT` | 真实实现（latent 字段运算） |
| `LoadImage` / `LoadImageMask` | `IMAGE`, `MASK` | 从文件加载图片/掩码 |
| `ImageScale` / `ImageScaleBy` | `IMAGE` | 缩放（OpenCV） |
| `ImageInvert` | `IMAGE` | 反色 |
| `ImageBlur` | `IMAGE` | 高斯模糊 |
| `ImageBatch` | `IMAGE` | 两图垂直拼接 |
| `ImagePadForOutpaint` | `IMAGE`, `MASK` | 外扩 padding（OpenCV） |
| `ImageCompositeMasked` | `IMAGE` | 掩码合成 |
| `ImageCrop` | `IMAGE` | 裁剪 |
| `ImageToMask` / `MaskToImage` | `MASK` / `IMAGE` | 图 ↔ 掩码转换 |
| `VAEDecode` / `VAEDecodeTiled` | `IMAGE` | 兼容节点（后端已完成 decode） |
| `VAEEncode` | `LATENT` | img2img：参考图编码 |
| `VAEEncodeForInpaint` | `LATENT` | inpainting：参考图 + 掩码 |
| `PreviewImage` / `SaveImage` | `IMAGE` / - | 预览 / 保存 |

### 采样与优化

| 节点 | 输出 | 说明 |
|------|------|------|
| `KSampler` | `LATENT` | 核心采样（sampler/scheduler/seed/cfg/steps/denoise） |
| `KSamplerAdvanced` | `LATENT`, `IMAGE` | 扩展参数 |
| `HiResFix` | `LATENT` | 高清修复（hires + FreeU/SAG/VAE Tiling + 后处理） |
| `ADetailer` | `IMAGE` | 局部重绘修复 |

### 模型增强 / 风格注入

| 节点 | 输出 | 说明 |
|------|------|------|
| `LORALoader` / `LoraLoader` / `LoraLoaderModelOnly` | `MODEL` | 加载 LoRA |
| `IPAdapterApply` | `MODEL` | IPAdapter 风格/人脸参考 |
| `CLIPVisionLoader` | `CLIP_VISION` | CLIP Vision（sd.cpp 原生） |
| `CLIPVisionEncode` | `CLIP_VISION_OUTPUT` | 透传（IPAdapter 直接吃图片路径） |
| `IPAdapterModelLoader` | `IPADAPTER` | IPAdapter（sd.cpp 原生） |
| `ControlNetLoader` | `CONTROL_NET` | ControlNet 模型 |

### CLIP 合并 / 旁路

| 节点 | 输出 | 说明 |
|------|------|------|
| `CLIPMergeSimple` / `CLIPMergeAdd` / `CLIPMergeSubtract` | `CLIP` | 真实实现（libtorch helper 合并 CLIP 权重段） |
| `LoraLoaderBypass` / `LoraLoaderBypassModelOnly` | `MODEL` | 等价 LoRA 加载 |

### 采样 / 模型配置

| 节点 | 输出 | 说明 |
|------|------|------|
| `ModelSamplingFlux` / `SD3` / `AuraFlow` | `MODEL` | 真实实现（映射 sd.cpp `flow_shift`） |
| `ModelSamplingContinuousEDM` / `ContinuousV` | `MODEL` | 真实实现（覆盖 denoiser sigma 区间；`discrete` 调度器忽略） |
| `ModelSamplingDiscrete` | `MODEL` | 真实实现（sd.cpp patch 强制预测类型 eps/v_prediction） |
| `ModelSamplingStableCascade` | `MODEL` | 透传（sd.cpp 按模型自动调度） |
| `ModelComputeDtype` | `MODEL` | 真实实现（映射 sd.cpp `wtype`，重载 ctx） |
| `ModelAttentionBackend` | `MODEL` | 部分实现（`flash_attn` → sd.cpp diffusion flash attention） |
| `RescaleCFG` | `MODEL` | 真实实现（sd.cpp patch 采样循环，1:1 复刻 ComfyUI） |
| `ModelNoiseScale` | `MODEL` | 真实实现（sd.cpp patch 缩放 ancestral 噪声） |
| `VideoLinearCFGGuidance` / `VideoTriangleCFGGuidance` | `MODEL` | 真实实现（按 batch/帧逐元素改 CFG scale） |

### Latent 序列化 / 保存

| 节点 | 输出 | 说明 |
|------|------|------|
| `LoadLatent` / `SaveLatent` | `LATENT` | `.latent` JSON 往返 |
| `CheckpointSave` / `VAESave` / `CLIPSave` / `ModelSave` / `ImageOnlyCheckpointSave` | - | 真实实现（libtorch helper 导出权重到 safetensors） |

### 模型合并 / 其他

| 节点 | 输出 | 说明 |
|------|------|------|
| `ModelMerge*`（20 变体） | `MODEL` | 真实实现（libtorch helper 合并 `diffusion_model.*` 权重，逐块 ratio） |
| `DiffusersLoader` / `unCLIPCheckpointLoader` / `ImageOnlyCheckpointLoader` | `MODEL`, `CLIP`, `VAE` | 加载器别名 |
| `ModelPatchLoader` | `MODEL_PATCH` | 返回名称 |
| `SVD_img2vid_Conditioning` | `CONDITIONING`×2, `LATENT` | 占位 |
| `QwenImageDiffsynthControlnet` / `ZImageFunControlnet` / `WanUni3CControlnetApply` / `AnimaLLLiteApply` / `SUPIRApply` / `USOStyleReference` / `ConditioningSetAreaPercentageVideo` | `CONDITIONING` | 透传 |
| `WebcamCapture` | `IMAGE` | 占位 |
| `ControlNetApplyAdvanced` | `CONDITIONING`×2 | 应用 ControlNet |
| `LatentUpscaleBy` | `LATENT` | 按倍数缩放 |
| `InpaintModelConditioning` | `CONDITIONING`×2, `LATENT` | inpainting 条件 |
| `PreviewAny` | `*` | 真实实现（序列化为字符串并打印） |

### 工具

| 节点 | 输出 | 说明 |
|------|------|------|
| `Reroute` | `*` | 真实实现（透传输入，即 Reroute 语义） |

> **注意**：StaticPy 无运行期自定义节点加载能力。新增节点需在 `comfycli/nodes.static.py` 中注册并重新编译。

## 技术架构

```
  workflow.json / --prompt --checkpoint
       │
  ┌────▼─────────────────────────────────────┐
  │  编排层（StaticPy 编译为机器码）          │
  │                                          │
  │  execution   节点 DAG 拓扑排序 + 输入链接解析│
  │  nodes       ComfyUI 节点定义（逐步补齐）  │
  │  cli_args    命令行参数解析               │
  │  main        CLI 入口                     │
  └────┬─────────────────────────────────────┘
       │ extern fn FFI 调用
  ┌────▼─────────────────────────────────────┐
  │  推理后端（stable-diffusion.cpp）          │
  │                                          │
  │  libsdcpp_adapter.so                     │
  │  ├── sd_pipeline_create / load / free      │
  │  ├── sd_pipeline_generate (txt2img)       │
  │  └── 内部封装：UNet/VAE/CLIP/Sampler/ControlNet/LoRA
  └────┬─────────────────────────────────────┘
       │
  ┌────▼─────────────────────────────────────┐
  │  GGML / CUDA / cuBLAS / cuDNN             │
  └──────────────────────────────────────────┘
```

StaticPy 只负责**编排**：解析 workflow JSON、拓扑排序、把节点输入解析为正确的 C API 参数、调用 sd.cpp 生成图片。所有张量计算都在 sd.cpp 内部完成。

### 编译流水线

```
comfycli/*.static.py  ──→  concat_src.py  ──→  _bundle.static.py
                                                │
                                                ▼
                    staticpy/static_translate.py  ──→  .ss (Scheme)
                                                │
                                                ▼
        staticpy/static_build_comfycli.sh  ──→  Chez AOT compile-file
        （拼接 prelude + stdlib + comfycli_ffi.scm + code）   │
                                                ▼
                         C launcher + objcopy + gcc  ──→  comfycli-bin (ELF)
```

> `staticpy/` 下的编译器核心（`static_translate.py` / `static_prelude.scm` / `static_stdlib.scm`）是 `/opt/ReScheme` 上游的**原样拷贝**；comfycli 特有的 FFI 与缺失内置放在 `comfycli/comfycli_ffi.scm`，构建脚本为 `staticpy/static_build_comfycli.sh`。详见 [BUILD.md](./BUILD.md)。
>
> `_bundle.static.py` 是 `concat_src.py` 生成的构建产物，**不入库**（`.gitignore`）；`build.sh` 会在缺失/过期时自动重建。

### 编排层健壮性

- **节点分发**：`call_node` 用 `NODE_GROUP` 查表路由到 5 个分类分派函数（clip/model/latent/image/misc），替代原先 200 分支的 `elif` 链。StaticPy 不支持一等函数值（无 `eval`/函数引用），故按类分组而非「字符串→函数」表。
- **失败传播**：节点失败返回 `(None, ...)`；`execute_prompt` 执行前用 `upstream_missing` 检测上游空产出，直接报出「哪个节点依赖哪个失败节点」，避免下游含糊崩溃。
- **无跨节点去重缓存**：早期缓存以 `class_type + 输入` 为键、不含节点身份，会跳过有副作用节点（`SaveImage` / 改 pipeline 状态的 `set_*`），已移除；`executed` 标记保证每节点只跑一次。
- **输入类型归一化**：`get_int/get_float/get_str` 经 `comfycli_ffi.scm` 的 `to_int/to_float/to_str` 归一化，workflow 里以字符串写的数字（`"20"`）不会在 FFI 边界静默错位。
- **`--prompt` 模式**不再硬编码 SDXL 的 `clip_l/clip_g.safetensors`；默认留空由 checkpoint 自带，可用 `--clip-l/--clip-g` 覆盖。

### 技术栈

| 层 | 组件 | 职责 |
|---|------|------|
| 源码语言 | StaticPy（Python 子集） | 无类继承 / lambda 闭包；异常用 if 守卫或 `guard` 语法糖；可直接 AOT 编译 |
| 编译器 | `static_translate.py` + Chez Scheme AOT | Python → Scheme → 机器码 |
| 推理后端 | `libsdcpp_adapter.so`（stable-diffusion.cpp） | UNet/VAE/CLIP/sampler/ControlNet/LoRA |
| 代码定位 | code search（my_db） | 语义搜索 + 调用链分析，辅助 1:1 翻译 |

## 核心优势

### 零 Python 依赖

```bash
# Python 版 ComfyUI — 需要
pip install torch torchvision ...  # ~2GB, 20+ 包
apt-get install ...                 # 系统依赖

# ComfyCLI — 只需要
LD_LIBRARY_PATH=./lib ./comfycli-bin workflow.json  # 单文件
```

- 不需要 Python 解释器
- 不需要 `pip install` 任何包
- 不需要虚拟环境
- 不需要 conda

### 二进制部署

```bash
# 产物列表
comfycli-bin            # 主二进制（ELF，包含所有编排逻辑）
comfycli-bin.so         # Chez AOT 编译产物
libsdcpp_adapter.so     # stable-diffusion.cpp 推理后端
lib/                    # GLIBC 兼容层 + 依赖运行时
```

- 单目录部署，`scp` 即用
- 不依赖系统 Python 版本
- Docker 镜像极小（alpine 兼容）
- 嵌入式 / 离线环境友好

### 编排层编译期优化

StaticPy 编译到机器码而非 CPython 字节码：

- 节点调度循环无 GIL 开销
- 函数调用为直接跳转而非 Python 属性查找
- 数据类型静态化，无装箱拆箱
- 启动时间≈进程启动时间，无 Python 模块导入开销

### 复用成熟推理后端

stable-diffusion.cpp 已经实现并验证：

- SD1.5 / SDXL / SD3 / FLUX 等多架构
- UNet 采样、VAE decode、CLIP encode
- LoRA、ControlNet、HiRes Fix、VAE tiling 等
- GGML 跨后端 + CUDA 高性能

StaticPy 不需要重新实现这些，只需要把它们包装成 ComfyUI 节点。

### code search 辅助翻译

用语义搜索定位 ComfyUI 对应源码，保证 1:1 翻译精度：

```bash
# 搜索 PromptExecutor 实现
cache_query "PromptExecutor execution loop" --repo /code/comfyui --type search

# 查看调用链上下文
cache_query PromptExecutor --repo /code/comfyui --type context --depth 2

# 搜索模型检测逻辑
cache_query "model config detect unet architecture" --repo /code/comfyui --type search
```

- 25k+ 代码块向量索引
- 4k+ 函数调用关系图
- 语义搜索而非关键词匹配

## 可行性

### 架构

分层架构（StaticPy 编排 → sd.cpp 推理 → GGML/CUDA）在生产项目中已有验证。
stable-diffusion.cpp 已经覆盖 UNet、VAE、CLIP、Sampler、ControlNet、LoRA 等核心推理操作，
编排层翻译是机械工作，不存在根本性技术障碍。

### 工作量

重写 ComfyUI 的 200+ 节点是数量问题而非难度问题。每节点：
- 用 code search 定位对应源码（秒级）
- 翻译为 StaticPy 的 `register_node` + 参数解析 + `sd_pipeline_*` 调用（分钟级）

复杂节点（多模型架构、采样器步进逻辑）需更多关注，但核心路径已由 sd.cpp 覆盖。

### code search 的作用

传统翻译的最大成本是手工在源码树中导航定位。本项目的 code search 系统（my_db）
已对 ComfyUI 做了全量语义索引（797 文件、25,586 chunks、4,017 函数、1,357 调用边）：

```bash
cache_query "model_config_from_unet" --type context --depth 3
  → 一次展示所有架构检测分支
  → 每个分支的 key pattern + 对应 config 结构
  → 调用链完整可见，无需手动翻文件
```

**源码定位成本接近于零，翻译变为纯粹的机械工作。**

### 对 StaticPy 的意义

ComfyUI 是 Python ML 生态中最复杂的纯推理项目之一：

- 200+ 节点、数十种模型架构（SD1.5 → SDXL → SD3 → FLUX）
- 动态 DAG 调度 + 显存管理 + 模型检测
- 依赖 torch、torchvision、transformers、scipy、Pillow 等数十个 pip 包

这个项目如果跑通，对 StaticPy 的推广有直接价值：**连 ComfyUI 这种体量的项目都能把编排层编译成单 ELF 零 Python 部署，其他 ML 项目只会更简单。** 它证明 StaticPy 不是玩具语言，而是能承载生产级推理引擎编排的编译工具。同时验证"Python 写编排 + C++ 写计算"的混合编译模式在大型项目中的可行性。

## 开发进度

**活跃模块**（进入 `_bundle.static.py` 并被编译）：

```
[x] cli_args.static.py        CLI 参数解析
[x] sd_backend.static.py      stable-diffusion.cpp C API FFI 封装（extern fn）
[x] nodes.static.py           132 个节点定义（覆盖 ComfyUI 全部 120 个节点名）
[x] execution.static.py       DAG 拓扑排序 + 输入链接解析
[x] main.static.py            CLI 入口（workflow JSON / --checkpoint --prompt）
[x] comfycli_ffi.scm          共享库加载 + 上游缺失内置
```

**已由 sd.cpp 后端替代**（torch 时代模块，源码保留但**不再进 bundle**）：

```
[~] folder_paths / comfy_types / supported_models* / model_detection
[~] model_sampling / latent_formats / model_base / model_management
[~] sd / clip_model / controlnet / sample / lora / k_diffusion
    — 模型检测、采样调度、显存管理、LoRA/ControlNet 全部由 stable-diffusion.cpp 内部承担
```

**端到端验证**：

```
[x] workflow SDXL → 图片（1024×1024，sd_xl_base_1.0 + clip_l/clip_g）
[x] --prompt 命令行模式
[x] HiResFix 2560×1440 / GGUF（Z-Image + Qwen LLM）/ IPAdapter / LoRA / ADetailer
[x] img2img（VAEEncode + denoise）/ inpainting（VAEEncodeForInpaint + mask）
[x] ControlNet 流程 / ImageScale / ImageInvert / EmptyImage
[x] 执行引擎 validate（未知节点/非法链接/环检测）
[x] 部署包 GPU 57MB / CPU 35MB（零 Python、零 pip、无 ONNX）
```

## 局限

- 不支持 Python 动态特性（类继承、lambda 闭包、生成器、`eval`/`exec`、运行期 `import`）—— ComfyUI 核心编排逻辑均不需要
- 异常仅支持 `try/except` / `raise` / `assert` 语法糖（`guard`），非完整语义
- 无自定义节点动态加载——自定义节点需编译期注册
- CLI 先行，无 WebSocket/HTTP UI
- 同步执行，无 asyncio
- 已实现 132 个节点（ComfyUI 120 个内置节点名全覆盖）；核心链路真实实现，其余受 sd.cpp 能力边界限制（详见「对齐度说明」）
- 执行引擎为简化版：拓扑排序 + 校验 + 输出缓存，但无 list 输入广播 / lazy 求值 / ExecutionBlocker / 子图
- 数据类型为占位（LATENT/CONDITIONING/MODEL 非真实张量），节点间无法做张量级操作

## 项目文件

### 文档

| 文件 | 说明 |
|------|------|
| [设计文档](./design.md) | 技术架构、模块映射、翻译策略、工程顺序 |
| [编译流水线](./BUILD.md) | 本地编译、增量编译、编译产物说明 |
| [部署文档](./deploy.md) | 纯二进制部署、GLIBC 兼容方案、远程要求 |
| [远程 GPU 部署](./remote_server.md) | Xiangongyun 实例；路线 A comfycli-bin workflow 一键 / 路线 B img_hires 出图管线手动 scp 清单 / 路线 C sd-cli 图片编辑（edit.sh） |
| [ComfyUI 分析报告](./comfyui_analysis.md) | code search 语义索引结果 (797 文件, 25,586 chunks) |
| [code search 使用文档](https://github.com/quqiufeng/my_db/blob/main/coding.md) | 语义搜索 + 向量查询工具用法 |

### 脚本

| 脚本 | 说明 | 常用命令 |
|------|------|---------|
| [`./build.sh`](./build.sh) | 编译 ELF 二进制 + C++ `.so` | `./build.sh` |
| [`./deploy.sh`](./deploy.sh) | 打包依赖 `.so` + GLIBC 兼容层 + SCP | `GLIBC_TARGET=2.35 ./deploy.sh --scp user@remote_host` |
| [`./concat_src.py`](./concat_src.py) | 按依赖顺序合并 `comfycli/*.static.py` 为 `_bundle.static.py` | `python3 concat_src.py` |

### 目录

| 目录 | 说明 |
|------|------|
| [`staticpy/`](./staticpy/) | StaticPy 工具链（上游 `/opt/ReScheme` 原样拷贝）+ `static_build_comfycli.sh` 构建胶水 |
| [`comfycli/`](./comfycli/) | StaticPy 编排层源码（节点、DAG、CLI）+ `comfycli_ffi.scm`（FFI/内置） |
| [`cpp/sd/`](./cpp/sd/) | stable-diffusion.cpp 适配器 (`sdcpp_adapter.h/.cpp` + build 脚本) |

### 参考

- [ComfyUI 源码](https://github.com/comfyanonymous/ComfyUI)
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- [Chez Scheme (Cisco)](https://github.com/cisco/ChezScheme) — Apache 2.0 开源，AOT 编译后端
