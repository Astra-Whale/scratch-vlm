# 图像 Captioning 性能 Baseline 全谱坐标系

**Date**: 2026-07-14
**Owner**: 徐悦
**Purpose**: 为 scratch-vlm 建立 captioning 性能对标的**完整坐标系**(数据集 × 指标 × 时代),把本项目 **Flickr8k 标准 test corpus BLEU-4 = 31.76%** 诚实地放进去,并回答"是否应补报 CIDEr/METEOR/SPICE"。

> **与已有文档的分工(避免重复)**:
> - `docs/competitor_benchmark.md` 已覆盖**同类极简 / 端侧项目定位**(nanoVLM / TinyLLaVA / MobileVLM / Moondream)和 corpus-vs-sentence BLEU 铁律 —— 本文不重复,只做**增量**。
> - 本文增量 = ①三大数据集规模表 ②五大指标对比表(重点讲 **CIDEr 是现代主指标、BLEU-4 已相对过时**)③**按时代的模型 × 数据集 × 指标大表(补齐 CIDEr 维度)** ④ 31.76% 的坐标定位 ⑤补 CIDEr 的成本/价值建议。
> - **诚实原则**:所有数字带来源 URL;二手/转引或拿不准的一律标 **[待核]**;跨数据集 / 跨 tokenizer / corpus-vs-sentence **不可直接比较**。

---

## 零、TL;DR(先读这个)

1. **现代 captioning 主指标是 CIDEr(不是 BLEU-4)**。2015 年 CIDEr 提出后,所有主流 leaderboard(COCO Karpathy)以 CIDEr-D 为**首要排序键**,BLEU-4/METEOR/ROUGE/SPICE 为辅。BLEU-4 仍报,但**已不是 SOTA 竞争维度**。
2. **我们只报了 BLEU-4,没报 CIDEr —— 相对现代坐标系是"缺了主轴"**。这不是错误(BLEU-4 依然是合法的 corpus 指标),但对标现代工作时"不在同一根轴上"。
3. **31.76% 是 Flickr8k 的 BLEU-4,不能和 COCO 的 40+ 直接比**:数据集不同、指标不同(BLEU vs CIDEr)、赛道不同(冻结主干 + 3.94M adapter vs 十亿级全量微调)。
4. **建议补 CIDEr/METEOR/SPICE(用 pycocoevalcap)** —— **零训练成本、纯 eval、预估半天**,价值高:让本项目的数字落到现代主轴上,面试时"我报了 CIDEr"比"我只有 BLEU-4"专业得多。

---

## 一、三大数据集(规模 / 标准 split / 参考数)

| 数据集 | 总图数 | 参考数/图 | 标准 split(Karpathy) train / val / test | 常用 test 规模 | 备注 |
|--------|--------|-----------|-------------------------------------------|---------------|------|
| **Flickr8k** | ~8,091 | **5** | 6,000 / 1,000 / 1,000 | 1,000 图 | 最小,入门/教学常用;现代 SOTA 少在此报 |
| **Flickr30k** | 31,783 | **5** | 29,783 / 1,000 / 1,000 | 1,000 图 | 中等;现代模型多做**零样本**评测 |
| **MS-COCO** | 123,287 | **5**(部分图更多) | **113,287 / 5,000 / 5,000** | 5,000 图(c5) | **事实标准**,几乎所有 SOTA 都在此报 |

**要点**:
- **Karpathy split**(Karpathy & Fei-Fei 2015)是三大数据集通用的标准划分,COCO 上 113,287 / 5,000 / 5,000 是所有论文的对标基准。
- **参考数都是 5**(和我们 Flickr8k 5 参考一致 —— 这点方法学上对齐了)。
- **COCO 有两套 test 协议**:Karpathy 离线 test(5,000 图,本地可算)和官方在线 server(c5=5 参考 / c40=40 参考)。论文表里 "(c5)/(c40)" 指后者。
- CIDEr 原作还提出 **PASCAL-50S / ABSTRACT-50S**(每图 50 参考)专门做"共识"评测,非主流训练集。

来源:
- 数据集规模与 Karpathy split:<https://www.kaggle.com/datasets/shtvkumar/karpathy-splits> · COCO Captions 官方:<https://arxiv.org/pdf/1504.00325>
- Karpathy split 原始定义:Karpathy & Fei-Fei, "Deep Visual-Semantic Alignments" (CVPR 2015)

---

## 二、五大指标对比(重点:CIDEr 是现代主指标,BLEU-4 已相对过时)

| 指标 (年份) | 原本用途 | 机制 | 量纲 | 是否 captioning 主指标 | 关键陷阱 |
|------------|---------|------|------|----------------------|---------|
| **BLEU-1/4** (2002) | 机器翻译 | n-gram **精度** + brevity penalty (BP) | 0–1 或 %(1-gram 高、4-gram 低) | ❌ **已相对过时**,仍报但非竞争维度 | ①corpus vs sentence 不等价 ②有无 BP 差很多 ③tokenizer(PTB vs regex)差 ±1-2 ④单参考 vs 多参考差异巨大 ⑤对词序不敏感 |
| **METEOR** (2005) | 机器翻译 | 精度+召回调和均值,含同义词/词干匹配 | 0–1 或 %(COCO 上 ~0.28–0.34) | 辅助 | 依赖 WordNet 同义词库;比 BLEU 更贴人判但仍 n-gram 系 |
| **ROUGE-L** (2004) | 自动摘要 | 最长公共子序列 F 值 | 0–1 或 %(COCO 上 ~0.56–0.60) | 辅助 | 只看召回结构,信息量低,几乎从不单独用 |
| **CIDEr / CIDEr-D** (2015) | **专为 captioning 设计** | **TF-IDF 加权 1–4 gram 余弦相似度**,先 stemming;IDF 压低跨图高频泛词、TF 突出该图显著词 | **通常 >1,以 100 为量纲**(COCO SOTA ~130–155) | ✅ **现代首要主指标** | ①量纲和 BLEU 完全不同(不要混淆 40 vs 130)②对词序敏感 ③会过度加权琐碎细节 ④需多参考才有意义 |
| **SPICE** (2016) | 专为 captioning 设计 | 解析成**场景图**(对象/属性/关系三元组)算 F 值,**语义级** | 0–1 或 %(COCO ~20–27) | 辅助(与 CIDEr 互补) | ①需依存句法解析 + Java ②对重复句处理差 ③与人类判断相关性最高(0.88 vs CIDEr 0.43 vs METEOR 0.53)但计算重 |

**核心结论(面试可讲)**:
- **为什么 CIDEr 取代 BLEU-4 成为主指标**:BLEU 源自机器翻译,只算 n-gram 精度、对"哪些词对这张图重要"无感;CIDEr 用 TF-IDF **突出该图独有的显著内容、压低"a/the/man"这类跨图泛词**,天然契合"一图多参考"的 captioning 场景,与人类判断相关性更高。所以 2016 年后论文表以 **CIDEr-D 为首要排序键**,并列时依次看 SPICE → METEOR → ROUGE → BLEU。
- **量纲陷阱(最容易翻车)**:BLEU-4 是 0–100% 尺度(SOTA 约 40),CIDEr 是以 100 为基准的尺度(SOTA 约 130–155)。**"我 31.76"和"SOTA 140"不是一个量纲,不能相减**。
- **SPIDEr** = CIDEr + SPICE 的线性组合,兼顾流畅度与语义,部分 RL 工作用它做 reward。

来源:
- CIDEr:<https://ar5iv.labs.arxiv.org/html/1411.5726>(arXiv 1411.5726, Vedantam et al. 2015)
- SPICE:<https://panderson.me/images/SPICE.pdf> · <https://link.springer.com/chapter/10.1007/978-3-319-46454-1_24>(SPICE vs CIDEr vs METEOR 与人判相关性 0.88/0.43/0.53)
- METEOR / ROUGE / BLEU 机制与陷阱:<https://arxiv.org/pdf/2008.12009>(NLG 指标综述)· COCO 评测协议:<https://arxiv.org/pdf/1504.00325>
- corpus vs sentence / 多参考陷阱:见 `docs/competitor_benchmark.md` 第三章"三条铁律"(不重复)

---

## 三、按时代的模型 × 数据集 × 指标大表(BLEU-4 / CIDEr 为主)

> **读表须知**:除注明外,BLEU-4/CIDEr 均为 **COCO Karpathy test、单模型**。"XE" = 交叉熵训练;"CIDEr-opt" = SCST/self-critical 强化学习优化 CIDEr 后(数值更高)。**跨数据集、跨指标不可直接比**。

### 3.1 经典时代(2015)—— 与本项目最可比的"起步量级"

| 模型 (年份) | 数据集 | BLEU-4 | CIDEr | 备注 |
|-----------|--------|--------|-------|------|
| **NIC / Show-and-Tell** (Vinyals 2015) | COCO | **27.7** | ~85–95 **[待核]** | CNN+LSTM,无 attention;当年 SOTA。原文主报 BLEU,CIDEr 早期常缺 |
| **Show-Attend-Tell** soft/hard (Xu 2015) | **Flickr8k** | **~19.5 / 21.3** | 未报 | 首个视觉 attention;**Flickr8k 上的经典锚点** |
| Show-Attend-Tell soft/hard | Flickr30k | ~19.1 / 19.9 | ~0.49–0.53(旧尺度)**[待核]** | |
| Show-Attend-Tell hard | COCO | ~25.0 | — | |

> **本项目直接对标锚点**:我们 Flickr8k 标准 test **corpus BLEU-4 = 31.76%**,同数据集 Show-Attend-Tell 是 **~19.5–21.3** → **明显超出**(约 +10 个 BLEU 点)。这是**同数据集、同指标(BLEU-4)、同参考数(5)** 的干净对比,可辩护性最强。

### 3.2 注意力 + 强化学习时代(2016–2018)

| 模型 (年份) | 数据集 | BLEU-4 | CIDEr | 备注 |
|-----------|--------|--------|-------|------|
| **SCST / Att2all** (Rennie 2017) | COCO | 34.2 | **114.0** | 首个用 RL 直接优化 CIDEr;把在线 server CIDEr 从 104.9→114.7 |
| **Up-Down (bottom-up)** (Anderson 2018) XE | COCO | 36.2 | 113.5 | Faster R-CNN 区域特征 + attention;赢 2017 VQA Challenge |
| **Up-Down** CIDEr-opt | COCO | **36.3** | **120.1** | 此后多年的标准 baseline 行 |

### 3.3 Transformer 时代(2019–2020)

| 模型 (年份) | 数据集 | BLEU-4 | CIDEr | 备注 |
|-----------|--------|--------|-------|------|
| **AoANet** (Huang 2019) | COCO (CIDEr-opt) | 38.9 | **129.8** | Attention-on-Attention |
| **M² Transformer**(Meshed-Memory, Cornia 2020) | COCO (CIDEr-opt) | 39.1 | **131.2** | 记忆增强 + 网状连接 |
| **X-Transformer / X-LAN**(Pan 2020) | COCO (CIDEr-opt) | 39.7 | **132.0–132.8** | X-Linear attention;无 ensemble 当时最佳 |
| (后续) DLCT / PureT 等 | COCO | ~40–41 | ~133–136 | 榜单持续爬升 |

### 3.4 视觉-语言预训练(VLP)时代(2020–2022)

> 下列为 **COCO Karpathy 单模型**(LAVIS 官方汇总,按 CIDEr 排序;多为 CIDEr-opt / 大数据):

| 模型 (年份) | BLEU-4 | CIDEr | METEOR | SPICE | 备注 |
|-----------|--------|-------|--------|-------|------|
| **OSCAR** (2020) | 40.7 | 140.0 | 30.6 | 24.5 | 引入 object tags 对齐 |
| **VinVL** (2021) | 41.0 | 140.9 | 31.1 | 25.2 | 强化视觉特征;在线 server 首超人类 CIDEr |
| **BLIP** (2022) | 40.4 | 136.7 | 31.4 | 24.3 | bootstrapping 图文预训练 |
| **SimVLM** (2021) | 40.6 | 143.3 | 33.7 | 25.4 | prefix-LM,13× 更多数据 |
| **CoCa** (2022) | 40.9 | 143.6 | **33.9** | 24.7 | 对比+captioning 联合;METEOR 领先 |
| **LEMON** (2021) | 42.6 | 145.5 | 31.4 | 25.5 | 大规模检测器预训练 |
| **OFA** (2022) | **44.9** | **154.9** | 32.5 | **26.6** | 统一多任务框架;此表 CIDEr 榜首 |
| (对照) BUTD baseline | 36.5 | 113.5 | 27.0 | 20.3 | 即 Up-Down 量级 |
| (对照) ClipCap | 32.2 | 108.4 | 27.1 | 20.1 | 轻量 CLIP+GPT2,和我们思路最像 |

> **注意 ClipCap**:CLIP 特征 + 冻结 GPT-2 + 轻量 mapping,是和**我们架构最接近**的参照(冻结主干 + 小 adapter),COCO 上 BLEU-4 32.2 / CIDEr 108.4 —— 但它训在 COCO 全量、我们训在 Flickr8k,**仍不可直接比**。

### 3.5 LLM-VLM 时代(2023–2025)

> **重要**:这些模型在论文里**多以 VQA / MME / MMBench / MMMU 等综合 benchmark 为主战场**,captioning BLEU/CIDEr 常只作辅助报告甚至不报。以下 COCO CIDEr 多为 fine-tuned:

| 模型 (年份) | 参数 | COCO CIDEr | 其他 | 备注 |
|-----------|------|-----------|------|------|
| **Flamingo** (2022) | 80B | 138.1 | — | few-shot;规模巨大但 caption 反不及专用模型 |
| **GIT / GIT2** (2022) | 0.7B / 5.1B | 144.8 / 145.0 | NoCaps ~124 | 单塔生成式 |
| **BLIP-2** (2023) | (Q-Former<190M 可训) | **145.8**(FT) | NoCaps 零样本 121.6 | 冻结图像塔 + 冻结 LLM + Q-Former,**理念与我们最近** |
| **BEiT-3** (2022) | 1.9B | 147.6 | — | |
| **PaLI-17B** (2022) | 17B | **149.1** | NoCaps 124.4 | 无 CIDEr-opt 的最高分 |
| **Qwen-VL** (2023) | 9.6B | 131.9 **[待核]** | B4 39.1 / METEOR 30.1 | 二手转引,原文以 VQA 为主 |
| **LLaVA-1.5** (2023) | 7.3B | 133.7 **[待核]** | B4 39.4 / METEOR 29.5 | 二手转引;LLaVA 原论文**未把 caption CIDEr 作主指标** |

> **LLM-VLM 时代的关键 insight**:①**赛道已从"刷 caption 榜"转向"综合多模态能力"**(VQA / OCR / 推理 / grounding),很多顶级 VLM 甚至不报 COCO caption CIDEr;②**BLIP-2 的理念(冻结图像塔 + 冻结 LLM + 只训桥接模块 <190M)与我们几乎一致** —— 这是我们可以强攀的"血缘",区别只是规模和数据。

来源(第三章):
- Show-and-Tell:<https://arxiv.org/abs/1411.4555> · Show-Attend-Tell:<https://arxiv.org/pdf/1502.03044>
- SCST:<https://arxiv.org/pdf/1612.00563> · Up-Down:<https://arxiv.org/pdf/1707.07998>
- M² Transformer:<https://arxiv.org/pdf/1912.08226> · X-Transformer/X-LAN:<https://arxiv.org/pdf/2003.14080> · AoANet(ICCV 2019, Huang et al.)· S²-Transformer 汇总表:<https://www.ijcai.org/proceedings/2022/0224.pdf>
- VLP 大表(OSCAR/VinVL/BLIP/SimVLM/CoCa/LEMON/OFA/BUTD/ClipCap):<https://github.com/salesforce/LAVIS/blob/main/dataset_card/coco_caption.md>
- BLIP:<https://arxiv.org/pdf/2201.12086> · VinVL:<https://arxiv.org/pdf/2101.00529> · CoCa:<https://research.google/blog/image-text-pre-training-with-contrastive-captioners/>
- BLIP-2:<https://arxiv.org/html/2301.12597> · GIT:<https://arxiv.org/pdf/2205.14100> · PaLI(含 Flamingo/GIT2/BEiT-3 对比表):<https://arxiv.org/pdf/2209.06794>
- Qwen-VL / LLaVA-1.5 caption CIDEr(二手转引,**[待核]**):见 <https://arxiv.org/pdf/2310.08825>(From CLIP to DINO,含 COMM 对比表)
- 时代演进(平均 BLEU-4:全局 CNN 25.1 → attention 35.3 → self-attn 40.0 → VLP 峰值 42.6):<https://arxiv.org/pdf/2107.06912>(From Show to Tell survey)

---

## 四、把本项目 31.76% 放进坐标系(诚实定位)

### 4.1 我们的数字落在哪

- **数据集**:Flickr8k 标准 test(1,000 图,5 参考)
- **指标**:corpus BLEU-4 = **31.76%**(n-gram 0.762 / 0.457 / 0.242 / 0.121,带 BP,regex 分词)
- **训练**:从零训练于 Flickr8k train(5,999 图),train/test 不相交、无泄漏
- **架构**:冻结 CLIP-ViT-L/14@336 + 冻结 Qwen2.5-0.5B-Instruct + 只训 3.94M MLP projector(0.49%)

### 4.2 与谁可比 / 不可比(一张判断表)

| 对比对象 | 可比? | 理由 |
|---------|-------|------|
| **Show-Attend-Tell @ Flickr8k(~19.5–21.3 B4)** | ✅ **最可比** | 同数据集、同指标(B4)、同参考数(5)。我们 31.76 **明显超出** |
| Show-Attend-Tell @ Flickr30k/COCO | ⚠️ 半可比 | 同指标不同数据集,只能说"同量级/超出",不能声称追平 |
| COCO SOTA BLEU-4(40–45) | ❌ 不可比 | **不同数据集**(COCO≫Flickr8k)+ 十亿参数全量微调 + 百万级数据 |
| **任何 CIDEr 数字(114–155)** | ❌ **完全不可比** | **不同指标 + 不同量纲**。CIDEr 以 100 为基准,BLEU-4 以 100% 为基准,**不能相减/类比** |
| ClipCap / BLIP-2(冻结主干思路) | ⚠️ 血缘可比 | 架构理念同源(冻结+小 adapter),但数据集/规模差太多,只能讲"同一方法论谱系" |

### 4.3 四条最该注意的对标陷阱(针对我们)

1. **量纲陷阱**:别把 31.76(BLEU-4)和 SOTA 的 140(CIDEr)放一起 —— 不是一根轴。
2. **数据集陷阱**:31.76 是 Flickr8k,COCO 的 40+ 不能直接压过来(COCO 大 15×、难度分布不同)。
3. **tokenizer 陷阱**:我们用 regex 分词,论文标准是 PTBTokenizer(pycocoevalcap),**同一模型可差 ±1–2 BLEU 点** —— 已在 README 诚实标注。
4. **赛道陷阱**:我们是冻结主干 + 3.94M adapter + 0.5~几个 epoch;SOTA 是全量微调 + 十亿参数 + 百万级数据。**"数字差距 ≠ 能力差距,是做的另一件事"**。

### 4.4 面试如何诚实陈述(话术)

> "我报的是 **Flickr8k 标准 test 的 corpus BLEU-4 = 31.76%**,方法学干净:从零训练、5 参考、train/test 无泄漏。同数据集上经典的 Show-Attend-Tell 大约 19–21,我明显超出 —— 这靠的是站在 CLIP-L + Qwen2.5 预训练主干肩上、只训 3.94M projector,正好证明'现代预训练 + 微型 adapter'的杠杆。
>
> 我很清楚**现代 captioning 主指标是 CIDEr 不是 BLEU-4**,而且 COCO SOTA 的 CIDEr 在 130–155 —— 但那是不同数据集、不同指标、十亿参数全量微调,和我'冻结主干 + 0.49% 可训'不是一个赛道。BLEU-4 在我这里是'对齐已学到'的验证信号,不是 KPI;我的交付物是端侧可部署性(1.6GB 推理显存、Orin 兼容、量化曲线)。
>
> 如果需要 apples-to-apples 对标现代工作,我可以用 pycocoevalcap 补报 CIDEr/METEOR/SPICE —— 纯 eval、半天就能出。"

---

## 五、是否补报 CIDEr / METEOR / SPICE(建议:补,优先级中)

### 5.1 结论:**建议补,用 `pycocoevalcap`,优先级排在量化/迁移之后但值得做**

### 5.2 成本

| 项 | 成本 | 说明 |
|----|------|------|
| 训练 | **0** | 纯 eval,不动模型、不用重训 |
| 依赖 | 低 | `pip install pycocoevalcap`(自带 BLEU/METEOR/ROUGE/CIDEr/SPICE)|
| 环境坑 | 中 | **SPICE 需要 Java(JRE)+ Stanford CoreNLP**;METEOR 也依赖 Java。CIDEr/BLEU/ROUGE 是纯 Python 无坑 |
| 工程量 | **~半天** | 把 evaluate.py 的生成结果转成 pycocoevalcap 的 `{img_id: [caption]}` 格式,喂 `COCOEvalCap` |
| 顺带收益 | — | pycocoevalcap 用 **PTBTokenizer**,可**同时消除 4.3 的 tokenizer 陷阱**,拿到与论文完全同工具链的 BLEU-4 |

### 5.3 价值

- **落到现代主轴**:CIDEr 是现代首要指标,报了它,本项目的数字才"在图上有坐标",而非只在 2015 年的 BLEU 轴上。
- **一次性四指标**:pycocoevalcap 一次跑出 BLEU-1/4 + METEOR + ROUGE-L + CIDEr + SPICE,**双报都省了**。
- **可辩护性**:面试被问"为什么不报 CIDEr"时,"我报了"远胜于"我只有 BLEU-4"。
- **顺带修 tokenizer 陷阱**:换成 PTBTokenizer 后,BLEU-4 变成与论文严格同工具链,31.76 这个数就更硬。

### 5.4 落地建议

1. **最小动作(推荐先做)**:只接 pycocoevalcap 的 **CIDEr + BLEU(PTBTokenizer)** —— 纯 Python 无 Java 坑,半天内出;把 README 里 "corpus BLEU-4 31.76%" 升级成 "BLEU-4 xx.x / CIDEr xx.x(pycocoevalcap 标准工具链)"。
2. **完整动作(有余力再做)**:装 Java 补齐 METEOR + SPICE,凑齐现代五指标。
3. **预期**:Flickr8k 上 CIDEr 落点预估在 **60–90** 区间 **[待核,需实跑]**(Flickr8k 参考少、图简单,CIDEr 通常低于 COCO;不要拿去和 COCO 130+ 比)。
4. **诚实标注**:补报后仍须注明"Flickr8k ≠ COCO,CIDEr 不跨数据集比"。

---

## 六、给 ROADMAP 的一句话反哺

- **补 CIDEr 是"低成本高性价比"项**:不烧 GPU、半天工程,却能把项目从"只有 2015 年 BLEU 轴"拉到"现代 CIDEr 主轴",建议插入到量化(P2)之后、作为一个独立小任务(P2.5)。
- **对标叙事定稿**:对外一律讲"Flickr8k 标准 test、同数据集超越 Show-Attend-Tell、现代主指标是 CIDEr(我可补报)、我的赛道是端侧全流程不是刷榜" —— 这套话术准确、可辩护、且主动暴露了对现代指标体系的认知(加分)。
