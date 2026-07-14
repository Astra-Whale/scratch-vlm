# 竞品与 Benchmark 定位调研

**Date**: 2026-07-13
**Purpose**: 为 scratch-vlm 项目(面向 DJI 端侧 AI 系统工程师岗)做同类项目定位 + captioning benchmark 参考量级调研,产出"诚实定位话术 + 常见质疑应对",避免被"BLEU 偏低 / 数据集太小"将军。
**原则**:本项目定位是**端侧 AI 全流程 demo(训练→推理→量化→硬件迁移)**,不是刷 captioning 榜。以下所有数据均注明来源,跨数据集 / 跨 corpus-vs-sentence 不可直接比较。

---

## 一、核心结论(先读这个)

1. **同类项目分两派,我们属于"教学/极简派"而非"SOTA 派"**:HuggingFace nanoVLM 明确声明"不是新 SOTA,是教育性作品";我们和它同一定位区间(冻结主干 + 极小可训 adapter + 可读代码),但**多做了端侧全流程(量化 + Jetson 迁移分析)**——这正是我们相对 nanoVLM 的差异化。
2. **"冻结主干 + 只训 projector"是业界公认的标准 alignment 阶段做法**,不是偷懒。TinyLLaVA / MobileVLM / LLaVA 都在对齐阶段冻结视觉塔、只训 projector,我们把这个阶段做到"可复现 + 可解释"是加分项。
3. **BLEU-4 不该拿来跟 SOTA 比,而该跟"经典模型 + 同数据集量级"比**。我们 Flickr1K corpus BLEU-4 20.59% 与 Show-Attend-Tell 在 Flickr30k 的 ~19-20 同量级;现代 SOTA(OFA/BLIP-2)在 COCO 是 42-44,但那是**十亿级参数 + 百万级数据 + 全量微调**,与我们"3.94M 可训 / 900 张训练图 / 冻结主干"完全不是一个赛道。
4. **诚实定位一句话**:"我不是在刷 captioning 榜,而是用最小代价跑通端侧 VLM 全流程。BLEU 只是证明 vision→language 对齐真的学到了(相对随机基线 +24×),我的交付物是**可部署性证据**——1.6GB 推理显存、Orin 兼容审计、量化曲线,这些才是端侧系统岗关心的。"
5. **对"数据集太小"的最强应对**:承认它,并转化为叙事——"小数据 + 冻结主干恰恰是端侧 adapter 经济学的体现;我留了 scale-up 路径(S1 标准 split),但优先级低于量化/迁移,因为岗位要的是系统能力不是榜单名次。"

---

## 二、同类"极简 / 端侧 VLM"项目定位对比

| 项目 | 参数规模 | 主干选型 | 训练策略 | 核心卖点 / 定位 | 报的指标 |
|------|---------|---------|---------|----------------|---------|
| **HuggingFace nanoVLM** | 222M (SigLIP-B/16 + SmolLM2-135M) | 预训练视觉塔 + 小 LLM | 纯 PyTorch,750 行,模块化;projector ~50 行 | **教学/极简**:"最简单、最快训练小 VLM 的仓库";作者明确"**不声称是新 SOTA**,是教育性作品" | MMStar 35.3%(单 H100 训 6h / 1.7M 样本) |
| **TinyLLaVA** | 1.5B / 2B / 3.1B (SigLIP + Phi-2/TinyLlama) | 预训练视觉塔 + 小 LLM | 冻结视觉塔 + 训 projector + (阶段性)训 LLM | **"小胜大"**:3.1B 变体在综合 benchmark 上超过 7B LLaVA-1.5 | VQAv2 / POPE / 综合 VLM benchmark 准确率 |
| **MobileVLM** | 1.4B / 2.7B (MobileLLaMA 主干) | 自研 mobile-scale LLM + CLIP | 视觉指令微调阶段冻结 LLM,只更新 LoRA(1.4B 仅 8.87% / 2.7B 仅 7.41% 参数) | **端侧速度**:第一个"从零、可复现、面向移动端"的 VLM;Snapdragon 888 上 2.7B 比 OpenLLaMA-3B 快 2× 且省 2/3 RAM | 标准 VLM benchmark + **on-device tok/s** |
| **MobileVLM V2** | 1.7B / 3B / 7B | 同上 + 轻量 projector | 高质量数据 + 学术数据 | **轻量 projector + 低延迟**:1.7B 匹敌 3B;**在 NVIDIA Jetson AGX Orin 上延迟低于同量级** | 标准 benchmark + Jetson Orin 延迟 |
| **Moondream (moondream2)** | 0.5B / 1.6B (SigLIP + Phi-1.5, LLaVA 数据) | 预训练视觉塔 + Phi | LLaVA 式训练 | **极致端侧部署**:CPU / 手机 / 树莓派可跑;0.5B 是"极端边缘部署的蒸馏目标,每 MB 都重要" | 通用 VLM 能力 + 边缘可运行性 |
| **本项目 scratch-vlm** | ~801M (CLIP-L/14@336 + Qwen2.5-0.5B),**仅 3.94M / 0.49% 可训** | LLaVA v1.5 官方选型 | **冻结视觉塔 + 冻结 LLM,只训 2 层 MLP projector** | **端侧 AI 全流程 demo**(训练→推理→量化→Jetson 迁移分析);推理显存 1.6GB,Orin 兼容审计通过 | corpus + sentence BLEU-4(双报)+ **推理显存 + 迁移可行性** |

**定位坐标读法**:
- **横轴 = 榜单野心**:TinyLLaVA(刷综合 benchmark)↔ nanoVLM / 本项目(不刷榜、讲工程/教学)。
- **纵轴 = 端侧程度**:MobileVLM / Moondream(真机测速)↔ 本项目(显存 + 迁移分析,量化在做)。
- **我们的独特格**:同 nanoVLM 一样是"极简可读、不刷榜"派,但**比 nanoVLM 多了端侧全流程叙事**(量化 + Jetson 迁移);比 MobileVLM/Moondream 参数小得多、可训比例(0.49%)极端小,主打"最小 adapter + 冻结主干经济学"。

来源:
- nanoVLM: <https://github.com/huggingface/nanoVLM> · <https://www.marktechpost.com/2025/05/08/hugging-face-releases-nanovlm-a-pure-pytorch-library-to-train-a-vision-language-model-from-scratch-in-750-lines-of-code/>
- TinyLLaVA: <https://arxiv.org/abs/2402.14289> · <https://arxiv.org/html/2402.14289v1>
- MobileVLM: <https://arxiv.org/html/2312.16886v1> · MobileVLM V2: <https://www.emergentmind.com/papers/2402.03766>
- Moondream: <https://moondream.ai/p/models> · <https://github.com/m87-labs/moondream>

---

## 三、图像 captioning benchmark 参考量级

> **三条铁律(应对质疑用)**:
> 1. **corpus BLEU ≠ sentence BLEU**:corpus 先在全数据集上汇总 n-gram 命中数再算精度和 brevity penalty;sentence 是逐句算再平均。二者数学上不等价,**sentence 版(尤其带平滑)在短 caption 上数值偏高**。论文报 SOTA 几乎都用 corpus BLEU。
> 2. **跨数据集不可比**:COCO / Flickr30k / Flickr8k 难度、reference 数量不同,BLEU 不能横跨数据集比。
> 3. **reference 数量影响巨大**:同一 corpus 用不同数量 reference 算出的 BLEU 也"极具欺骗性",不可直接比。

### 3.1 经典模型(与本项目同一"起步量级")

| 模型 (年份) | 数据集 | BLEU-4 | 类型 | 备注 |
|-----------|--------|--------|------|------|
| Show and Tell / NIC (Vinyals 2015) | **COCO** | **27.7** | corpus | 当年 SOTA;CNN-encoder + RNN-decoder |
| Show and Tell / NIC | Flickr30k | (论文主报 BLEU-1 = 66) | — | 原文 Flickr30k 主要报 BLEU-1,BLEU-4 未强调 |
| Show, Attend and Tell — soft attn (Xu 2015) | Flickr30k | **~19.1** | corpus | 首个视觉 attention captioning |
| Show, Attend and Tell — hard attn | Flickr30k | **~19.9** | corpus | |
| Show, Attend and Tell — soft/hard | Flickr8k | ~19.5 / 21.3 | corpus | |
| Show, Attend and Tell — hard attn | COCO | ~25.0 | corpus | |

> **本项目锚点**:Flickr1K **corpus BLEU-4 20.59%** ≈ Show-Attend-Tell 在 Flickr30k 的 ~19-20 量级。**但请注意**:数据集不同(Flickr1K 只有 900 训练图 vs Flickr30k 约 3 万图),所以只能说"**数字同量级**",不能声称"追平/超越"。这个对标的意义是:**证明只训 3.94M projector 就能达到经典 attention 模型的输出质量水位**,是"站在 CLIP-L + Qwen 预训练肩上"的杠杆效应。

### 3.2 现代 SOTA(不同赛道,仅供说明差距来源)

| 阶段 / 模型 | 数据集 | BLEU-4 | 备注 |
|-----------|--------|--------|------|
| 全局 CNN 特征方法(平均) | COCO | ~25.1 | 早期 |
| Attention 方法(平均) | COCO | ~35.3 | |
| Self-attention 方法(平均) | COCO | ~40.0 | |
| 视觉-语言预训练 VLP(峰值) | COCO | ~42.6 | |
| **OFA / BLIP-2**(单模型 SOTA) | COCO Karpathy | **~43-44** | B@4 与 CIDEr 领先;十亿级参数 + 全量微调 |
| GIT | COCO Karpathy | 顶级但**非** Karpathy SOTA | 在 nocaps/TextCaps 等其他榜 SOTA |

> **差距归因(应对"你才 20,SOTA 都 44 了")**:SOTA 的 42-44 是 **①十亿级参数 ②百万~十亿级图文对 ③全量微调(不冻结) ④COCO 大数据集** 四者叠加的结果。我们**主动放弃**这四点中的每一个(参数 801M 但只训 0.49%、900 张训练图、冻结主干、Flickr1K),因为岗位要的是**端侧系统全流程能力**,不是榜单名次。差距不是"做得差",而是"**做的是另一件事**"。

来源:
- Show and Tell: <https://arxiv.org/abs/1411.4555> · <https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf>
- Show, Attend and Tell: <https://arxiv.org/pdf/1502.03044>
- corpus vs sentence BLEU: <https://thepythoncode.com/article/bleu-score-in-python> · <https://arxiv.org/pdf/2407.12832>
- BLEU 跨语料不可比警示: <https://www.digitalocean.com/community/tutorials/bleu-score-in-python>
- SOTA 量级与 BLEU-4 演进: <https://arxiv.org/pdf/2107.06912>(From Show to Tell survey) · <https://link.springer.com/article/10.1007/s00521-025-11672-x> · GIT: <https://ar5iv.labs.arxiv.org/html/2205.14100>

---

## 四、本项目的诚实定位话术

### 4.1 一句话定位(30 秒电梯版)
> "这是一个**端侧 VLM 全流程 demo**:LLaVA v1.5 风格,冻结 CLIP-L 视觉塔和 Qwen2.5-0.5B,只训 3.94M 的 MLP projector(占 0.49%)。重点不是刷 captioning 榜,而是打通**训练→推理→量化→Jetson 硬件迁移**这条端侧链路,并用硬数据证明可部署性——推理显存锁死在 1.6GB,过了 Orin NX 兼容审计。"

### 4.2 BLEU 数字怎么讲(2 分钟版)
> "BLEU 在这个项目里是**验证信号**不是**KPI**。我双报 corpus 和 sentence 两个版本:corpus BLEU-4 20.59% 用来跨论文对标(和经典 Show-Attend-Tell 在 Flickr30k 的 ~19-20 同量级),sentence 版 22.18% 带平滑、偏高、只做内部进度追踪。关键是相对随机初始化的 projector 基线(0.86%)提升了 **24 倍**——这证明 vision→language 对齐真的学到了。我很清楚现代 SOTA 在 COCO 是 42-44,但那是十亿参数 + 全量微调 + 大数据集,和我'只训 0.49% 参数 + 900 张图 + 冻结主干'不是一个赛道。"

### 4.3 价值主张(为什么这个 demo 对系统岗有意义)
1. **端侧 adapter 经济学**:3.94M 可训 / 801M 总参 = 0.49%。主体权重可离线一次性转换/量化,天然适配任何异构后端(Jetson / 自研 NPU)——这正是端侧部署最看重的性质。
2. **全流程覆盖 JD 关键词**:混合精度(bf16)、量化(GGUF Q4 / torchao,守 Orin 兼容红线)、异构迁移(4060→Orin→自研 NPU 三级路径)、算子/latency 分解(CLIP encode vs LLM decode 瓶颈)。
3. **诚信可复现**:声称=现实(模型本地真训过)、指标标清 corpus/sentence、非实测一律标"预估"、跨平台同一份代码零改动跑通。

---

## 五、常见质疑与应对(面试防御)

| 质疑 | 应对话术 |
|------|---------|
| **"BLEU 才 20,是不是效果很差?"** | "对标对象不是 SOTA,是经典模型和随机基线。相对随机 projector 提升 24×,和 Show-Attend-Tell 同量级。而且我只训了 0.49% 参数、0.5 个 epoch。这个数字是'对齐已学到'的证据,不是终点——我的交付物是端侧可部署性,不是 BLEU 分数。" |
| **"数据集才 900 张训练图,太小了。"** | "承认。这是**刻意的成本控制**:小数据 + 冻结主干正是端侧 adapter 经济学的体现,能用最小代价验证全流程。我留了 scale-up 路径(扩到 Flickr30k/COCO 标准 split),但优先级低于量化和迁移——因为这个 demo 服务的是'端侧系统岗',不是 captioning 竞赛。要 apples-to-apples 硬对标,我半天就能跑标准 split,但那不是当前重点。" |
| **"为什么冻结主干?是不是不会训 LLM?"** | "冻结是业界标准 alignment 阶段做法——LLaVA、TinyLLaVA、MobileVLM 在对齐阶段都冻结视觉塔只训 projector。CLIP 已经把图像对齐到语言语义,projector 只需学线性映射;冻结 LLM 保住通用能力、避免灾难性遗忘。这也让可训参数从数亿降到 3.94M,单卡几分钟就能迭代——对端侧和快速实验都是对的选择。" |
| **"为什么不用 Q-Former,用 2 层 MLP?"** | "LLaVA v1.5 消融证明 MLP projector 优于 Q-Former,参数少约 10×、训练成本低约 10×。端侧场景越简单越好,MLP 也更容易量化和跨后端迁移。" |
| **"你这和 nanoVLM 有啥区别?"** | "nanoVLM 是纯教学(明确声明不是 SOTA),我和它同属极简可读派,但我**多做了端侧全流程**:量化曲线、Jetson 三代硬件迁移分析、Orin 兼容审计、latency 分解。我的差异化不在模型精度,在**从训练到机载部署的完整工程链路**。" |
| **"corpus 和 sentence BLEU 为什么不一样?你在挑好看的报?"** | "恰恰相反,我两个都报并标清用途。corpus 是论文对标标准(先汇总 n-gram 再算),sentence 带平滑、在短 caption 上偏高,我只拿它做内部追踪。刻意双报是为了诚信——不藏、不挑。" |
| **"你这个能真跑在 DJI 机载上吗?"** | "现在证据链是:推理显存 1.6GB(Orin Nano 8G 都容得下)、bf16 权重对 Ampere 系 Orin 零转换、GGUF 硬件无关。真机 latency 我标注为'预估'(基于内存带宽 roofline),因为还没拿到 Orin DevKit。到大疆自研 NPU 是第三级迁移,需要目标 SDK,但我的冻结主干架构天然适合离线一次性转换——这是可迁移性的结构性优势。" |

---

## 六、对 ROADMAP 的建议(调研反哺)

- **S1(大数据集硬化)保持低优先级即可**:调研确认"数据集小"可用叙事化解,不必为堵嘴而烧 GPU 跑 COCO。若面试官强压,半天跑一个 Flickr30k 标准 split 即可 apples-to-apples。
- **强化"端侧全流程"差异化**:相对 nanoVLM 我们的护城河是量化 + 迁移分析(P2/P3),这是调研中最清晰的差异点,应在 pitch 里前置。
- **对标话术定稿**:README 已有的"corpus BLEU-4 与 Show-Attend-Tell Flickr30k ~19-20 同量级"表述**准确且可辩护**,建议保留;但务必每次都补一句"数据集不同,仅同量级参考"。
</content>
</invoke>
