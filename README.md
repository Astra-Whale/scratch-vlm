# scratch-vlm

散装 VLM 端侧部署项目 (Jetson Orin NX 目标平台)

面向 DJI 端侧 AI 系统工程师岗位定制的项目。参考 LLaVA v1.5 架构自己拼装
CLIP + MLP-Projector + LLM 的极简 VLM,重点展示端侧 AI 全流程能力
(训练 → 推理 → 量化 → 硬件迁移分析),仅训 projector、冻结视觉/语言主干。

**开发平台演进**:RTX 4060 Laptop 8G (Windows, baseline) → RTX 5060 Ti 16G
(Ubuntu, 升级实验平台)。同一份代码零改动跨两平台跑通;16G 仅用于提升训练
迭代效率,交付模型推理显存锁定在 8GB 内以保证 Orin/Jetson-class 可部署。

## 当前进度

- [x] **Week 1 · D1** · skeleton + 前向验证（2026-07-11, 4060）
- [x] **Week 2 · D2** · 训练闭环 + 推理（2026-07-12, 4060）
- [x] **Week 2 · D2.5** · 真数据 (Flickr1K 5K pairs) pretrain 300 步（2026-07-12, 4060）
- [x] **Week 2 · D2.6** · Train/Val split + BLEU-4 评测 + baseline 对比（2026-07-12, 4060）
- [x] **Week 2 · D2.7** · 迁移分析文档（[`benchmark/migration_analysis.md`](benchmark/migration_analysis.md)）
- [x] **迁移 · D3.0** · 迁到 Ubuntu/RTX 5060 Ti (Blackwell) · 3 项验证一致性（2026-07-13）
- [x] **升级 · D3.1** · 视觉塔 CLIP-B/32→L/14@336 + LLM SmolLM2→Qwen2.5-0.5B-Instruct · 重训（2026-07-13）
- [x] **量化 · D3.2** · torchao int8/int4 混合精度 (守 Orin 兼容) + 精度/体积曲线（2026-07-13）
- [x] **对齐 · D4.1** · LLM 换 **Qwen3-0.6B**（selfspec 选型）· `<think>` 剥离（2026-07-14）
- [x] **对齐 · D4.2** · **两阶段训练**:stage-1 projector 对齐 → stage-2 LoRA(q/v)联合 projector SFT（LLaVA-Instruct + 平衡 VQAv2）（2026-07-14）
- [x] **对齐 · D4.3** · **POPE 幻觉评测** avg F1 **78.59** + 数据配比消融链（2026-07-14）
- [x] **对齐 · D4.4** · **llama.cpp 全链路**:Qwen3 GGUF **Q4_K_M** 0.48GB/3.12× · llama-server SSE 流式 · fp16↔Q4 PPL（2026-07-14）
- [x] **对齐 · D4.5** · **mmproj 多模态集成**(CLIP+projector→llama.cpp,`llama-mtmd-cli` 端到端图文推理跑通)（2026-07-15）
- [ ] **对齐 · D4.6** · Xavier NX 实机部署(**不做**,硬件未到手 —— 保留 5060Ti baseline + 迁移分析)

> **selfspec 对齐**:见下方「selfspec 对齐结果」段与 [`ALIGN_SELFSPEC.md`](ALIGN_SELFSPEC.md) §0/§2、[`AGENT_CROSSCHECK.md`](AGENT_CROSSCHECK.md) §10。

## selfspec 对齐结果 · 两阶段训练 + POPE + llama.cpp(Qwen3-0.6B)

> 本段是按简历 selfspec 逐条补齐的对齐结果,LLM 全线统一为 **Qwen3-0.6B**(hidden 1024)。
> Flickr8k captioning 旗舰同样已迁到 Qwen3(见下方 SOTA 段,BLEU-4 **32.91** / CIDEr **94.0**,略胜旧 Qwen2.5 轨)。
> 旧 Qwen2.5 / SmolLM2 产物已归档至 `checkpoints/_archive_non_spec/`,不再作对标口径。硬数据与复核见 [`AGENT_CROSSCHECK.md`](AGENT_CROSSCHECK.md) §10。

### 两阶段训练(LLaVA v1.5 配方)

| 阶段 | 训练内容 | 数据 | 可训参数 |
|-----|---------|------|---------|
| **stage-1 · 对齐** | 仅 projector | Flickr8k(vision-language 对齐) | projector 4.20M |
| **stage-2 · 指令微调 SFT** | Qwen q_proj/v_proj **LoRA**(r=16/α=32)+ projector 联合 | LLaVA-Instruct(detail_23k + conversation_58k)+ **平衡 VQAv2 yes/no** | 6.49M(LoRA 2.29M + projector 4.20M) |

CLIP-L 与 Qwen3 base 全程冻结。stage-2 batch=1/accum=16(576 视觉 token + 长对话,峰值 7.2GB)。

### POPE 幻觉评测(#16)· `vlm_stage2_mix2`

| split | acc | precision | recall | f1 | yes% |
|-------|-----|-----------|--------|-----|------|
| random | 82.9 | 81.1 | 85.7 | **83.4** | 52.8 |
| popular | 76.1 | 71.9 | 85.7 | **78.2** | 59.6 |
| adversarial | 70.2 | 65.4 | 85.7 | **74.2** | 65.5 |
| **avg F1** | | | | **78.59** | |

> `random > popular > adversarial` 单调退化是 POPE 的**教科书级正确行为**;yes% 回到 ~53%(random)说明无 always-yes 偏置。0.6B 从零 VLM 对照 LLaVA-1.5-**7B** 的 ~85 F1,此量级体面。

### 两阶段/数据配比对幻觉的边际收益(#17)· 诚实消融链

| SFT 数据 | POPE 行为 | acc | avg F1 |
|---------|----------|-----|--------|
| 仅 detail(纯描述) | 全答 no(不会 QA) | 50 | 0 |
| +conversation | 全答 yes(训练短答 93% 是 yes) | 50 | 66.67(虚高) |
| **+平衡 VQAv2** | **均衡作答** | **70–83** | **78.59** |

> **结论**:模型幻觉/作答行为随训练数据分布走,**平衡 yes/no 是抑制过度肯定型幻觉的关键杠杆**。
> **诚信红线**:POPE 全程**零样本留出** —— 训练用 COCO **train2014**、POPE 用 **val2014**,图集无重叠;**从不**训练 POPE 同款"图里有没有 {物体}?"探针(那等于训练测试集)。yes/no 能力只来自**通用**平衡 VQA(VQAv2,LLaVA v1.5 正宗配方)。

### llama.cpp 端侧链路(#12/#13/#15/#18)

- **GGUF Q4_K_M**:Qwen3-0.6B → GGUF,Q4_K_M 分块量化 **0.48GB**(相对 f16 **3.12×** 压缩)。
- **llama-server SSE 流式**:HTTP + Server-Sent-Events 流式 token 输出跑通。
- **PPL 精度代价**:f16 **19.63** → Q4_K_M **21.35**(held-out,真 Q4_K_M via llama.cpp,非旧 torchao int4)。

详见 [`docs/llamacpp_pipeline.md`](docs/llamacpp_pipeline.md)。

---

### 5060 Ti (Ubuntu) 当前 SOTA · CLIP-L/14@336 + Qwen3-0.6B

**架构**:视觉塔 **CLIP-ViT-L/14@336px** (1024 维, 576 visual tokens),
LLM **Qwen3-0.6B** (selfspec 选型, hidden 1024)。CLIP-B/32+SmolLM2 与 Qwen2.5 均属早期轨,已归档
(`checkpoints/_archive_non_spec/`)。旗舰 projector = stage-1 对齐 ckpt `projector_stage1_qwen3_best.pt`。

| 组件 | 参数量 | 状态 |
|-----|--------|------|
| CLIP-ViT-L/14@336 (视觉塔) | ~304M | 冻结 |
| Qwen3-0.6B (LLM) | ~596M | 冻结 |
| MLP Projector (1024→2048→1024) | **4.20M** | 唯一可训 |
| 总 / 可训比 | ~904M / 4.20M | **0.46%** |

#### ✅ 干净可对标结果 · Flickr8k 标准 split(方法学正确)

**从零训练**于 Flickr8k train(5999 图),在**标准 Flickr8k 1000-test**(5 参考、train/test 不相交、无泄漏)评测:

**官方 `pycocoevalcap`(PTBTokenizer)全指标** —— 覆盖现代主指标 CIDEr:

| 指标 | 值 | 说明 |
|------|----|------|
| **CIDEr** | **0.940**(×100 惯例 94.0) | 🎯 现代 captioning 主指标 |
| **BLEU-4**(官方) | **32.91%** | 与本项目手写 corpus BLEU-4 **32.93% 几乎一致**(差 0.02,验证自写口径准确) |
| BLEU-1/2/3 | 75.8 / 59.6 / 44.9 | |
| METEOR / ROUGE-L | 0.276 / 0.573 | |

对比:Show-Attend-Tell Flickr8k BLEU-4 ~19.5(soft)/21.3(hard) → **我们 32.9 明显超出约 11-13 点**(同数据集/同指标/同 5 参考)。
> 归档对照:旧 Qwen2.5 轨同口径为 BLEU-4 31.73 / CIDEr 92.7;换 Qwen3(hidden 1024>896)略增容量后小幅提升。

> 这是本项目**方法学最干净、最可对标**的数字:proper Karpathy split、从零训练、5 参考、无 train-on-test,且用官方工具链复核。
> **诚实标注**:CIDEr **跨数据集不可比**(COCO SOTA CIDEr 130-155 是另一语料的 TF-IDF,不能与 Flickr8k 的 94.0 相减);
> 我们超越的是**同数据集的 2015 经典模型**,靠 CLIP-L+Qwen 预训练杠杆。全指标见 [`docs/benchmark_landscape.md`](docs/benchmark_landscape.md)。
> **为何超越 2015 经典模型**:站在 CLIP-L/14 + Qwen3 预训练主干肩上(冻结)、只训 4.20M projector —— 这正是项目论点(现代预训练 + 微型 adapter),不是刷 trick。
> 诚实标注:tokenizer 为 regex 分词(非 PTB),与论文工具链或差 ±1-2,不改变"超越经典 baseline"结论。

#### 架构开发期结果 · Flickr30k test-1k 子集(⚠️ 有 train-on-test,已被上方 Flickr8k 取代)

下表是升级 CLIP-L+Qwen-Instruct 时的开发记录,数据集为 Flickr30k 的 test-1k 子集内部 900/100 切分(**训练用到了标准 test 图**),仅作架构演进对照,**不作对标**:

**评测**（Flickr1K, 100 张 val, greedy, bf16）:

| 指标 | Baseline (未训 projector) | **Trained** | 提升 |
|------|--------------------------|-------------|------|
| **corpus BLEU-4**(与论文同法, 可对标) | 0.86% | **20.59%** | **+24×** |
| sentence BLEU-4(平滑, 内部追踪) | 2.99% | 22.18% | +7.4× |
| 平均生成长度 | 24.0 tok (发散) | 11.4 tok (贴近 GT 13.5) | 收敛 |

> **诚实标注**:(1) corpus BLEU-4 用于跨论文对标(Show-Attend-Tell 在 Flickr30k ~19-20);
> sentence-level 平滑版数值偏高、仅供内部进度追踪。(2) 数据集为 Flickr1K(900 train / 100 val),
> 规模远小于 COCO/Flickr30k, 数字只能作"同量级"参考, 非 leaderboard。(3) 能以 3.94M 可训参数
> (占 0.49%) 达到经典模型量级, 得益于站在 CLIP-L + Qwen 预训练主干的肩上——这正是端侧"冻结主干+
> 微型 adapter"经济学的体现。(4) ⚠️ **此 Flickr1K 实为 Flickr30k 的 Karpathy test split**,当前
> 900/100 是在该 test 集内部再切分——**训练用到了标准 test 图**,不符论文对标规范。**已由上方 Flickr8k
> 干净结果(S1)取代**:改用 Flickr8k proper split 从零训练、标准 test 评测,得可对标 corpus BLEU-4 31.76%。
> 此 20.59% 仅留作架构演进记录。

**显存**:推理 batch=1 峰值 **1.6 GB**(稳进 8GB, 连 Orin Nano 8G 可容);训练 ~9.9 GB
(batch=4 × grad-accum=4, 576 tokens 较重)。**推理显存 = 端侧可部署性的关键指标。**

**对比旧 SOTA**(4060, CLIP-B/32 + SmolLM2): corpus BLEU 提升自 ~17% 量级(见下方前期基线)。

### 迁移一致性 · 4060 (Windows) → 5060 Ti (Ubuntu)

同一份代码零改动跨两平台。旧配置(CLIP-B/32 + SmolLM2-360M, 300 步 batch=8)apples-to-apples:

| 测试 | 4060 Laptop (Windows) | 5060 Ti (Ubuntu) | 结论 |
|------|----------------------|-------------------|------|
| 训练 300 步 wall-time | 5.8 min (1165 ms/step) | **28.2 s (94 ms/step)** | 端到端 ~12×(注: 4060 为 IO-bound 读图; 5060 Ti 图走 page cache) |
| 训练 VRAM 峰值 (batch=8) | 2.67 GB | **2.63 GB** | 逐位一致 ✓ |
| 100 val BLEU-4 (4060 训的 ckpt 跨平台复评) | 17.15% (sentence, 原始记录) | **16.98% (sentence) / 14.60% (corpus)** | 跨平台推理一致 ✓ (差异为 transformers 版本生成细节) |

> **诚实标注**:BLEU 一致性用"同一个 4060 训的 checkpoint 在 5060 Ti 上复评"来证明(16.98% ≈ 原 17.15%)。
> 在 5060 Ti 上**重训**的 300 步(0.5 epoch)模型 BLEU 有 ±5 点方差属正常(短训练种子敏感),不影响迁移结论。

### 混合精度量化 (torchao weight-only · Orin 兼容)

只量化 Qwen 的 transformer Linear,**CLIP-L + projector + lm_head 守 bf16 → 混合精度**。100 val 实测:

| 量化 | corpus BLEU-4 | 权重常驻显存 | 说明 |
|------|--------------|-------------|------|
| bf16 (none) | 20.59% | 1536 MB | 基线 |
| **int8** weight-only | 20.04% | 1191 MB (-22%) | **近乎无损(±0.5 贪心噪声),直接采用** |
| int4 weight-only (g=128) | 18.33% | 1057 MB (-31%) | 2.3 BLEU 点换 31% 显存,内存极限时可选 |

> **关键发现(混合精度敏感度分析)**:Qwen `tie_word_embeddings=True`,量化 lm_head 会打破它与 embedding 的权重共享、凭空多一份副本(int8 少省 ~130MB),且 lm_head 是映射 15 万词表的敏感输出层——**排除 lm_head 一举两得:显存降幅 int8 从 -14%→-22%,int4 精度从 -4.6→-2.3 点**。
> **诚实标注**:(1) 整模型剩余最大 bf16 块是 CLIP-ViT-L/14(~608MB),要更深压缩下一步需量化 CLIP。(2) Orin 红线实战:torchao 0.17 默认 int4 packing 需非便携 `mslk` → 改用 `tile_packed_to_4d`(tinygemm,Ampere 原生)。(3) decode 提速与否 kernel 相关、未测。详见 [`docs/quantization_plan.md`](docs/quantization_plan.md)。

### 前期 4060 Laptop 基线硬数据（CLIP-B/32 + SmolLM2-360M, Windows, bf16, batch=4）

**模型规模**
| 组件 | 参数量 | 状态 |
|-----|--------|------|
| CLIP-ViT-B/32 (视觉塔) | 88M | 冻结 |
| SmolLM2-360M-Instruct (LLM) | 361M | 冻结 |
| MLP Projector (2 层, 1024→2048→960) | **3.54M** | 唯一可训 |
| 总 / 可训比 | 452.8M / 3.54M | **0.78%** |

**训练性能**（60 步 toy 数据）
| 指标 | 值 |
|-----|-----|
| Loss 下降 | **3.57 → 0.82 (-77%)** |
| Wall time | 11.8s (60 步) |
| 单步耗时 | **197 ms/step** |
| VRAM 峰值 | **1.45 GB** (卡容量 18%) |
| Optimizer | AdamW · lr 1e-3 · cosine + 3 步 warmup |

**推理效果**（未量化，temperature=0.0 greedy）
| 输入图 | VLM 输出 | 备注 |
|-------|---------|------|
| green triangle | `A green triangle.` | ✅ 形状+颜色都对 |
| orange square | `A green square.` | 形状对，颜色欠拟合（24 张 toy 数据下正常） |
| purple circle | `一个蓝色的圆。` | 形状对，中英切换 |

### D1 → D2 一句话总结

- ✅ CLIP + Projector + LLM 三段拼装 pipeline 完备
- ✅ ChatML 格式（Instruct 模型必需，裸 prompt 会秒 EOS）
- ✅ 只训 3.54M projector 参数 → toy 数据 loss 下降 77%
- ✅ Toy 生成结构对齐 chat 格式，形状识别正确（scale up 后颜色也会收敛）

### D2.5 · 真数据 pretrain 补充

**数据**：`nlphuji/flickr_1k_test_image_text_retrieval` — 1K 图 × 5 人工 caption = 5000 pairs

**训练**：300 步 · batch=8 · bf16 · AdamW lr=1e-3 · cosine schedule + 3 步 warmup

**结果**：
| 指标 | 值 |
|-----|-----|
| Loss 收敛 | **3.63 → 2.76 (-24%)** |
| Wall time | 5.8 分钟 |
| 单步耗时 | **1165 ms/step** (IO-bound, 从 disk 读图) |
| VRAM 峰值 | **2.67 GB** (batch=8, 8G 卡用 33%) |

**推理示例**（Flickr 真图 · greedy）：

| 图 | Ground truth (人写) | VLM 输出 |
|----|--------|---------|
| 男人+橙帽 | A man is wearing glasses and an orange hat. | `A man with a white shirt and a black hat is standing in front of a large building.` |
| 黑白狗草地 | A black and white dog is running through the grass. | `A dog is sitting on a mat.` |
| 亚裔小孩坐肩上 | A young asian child sitting on parents' shoulders, clapping. | `A young girl with a red hat and a blue dress is standing in front of a red and white striped flag.` |

**分析**：这是**典型 pretrain-only 阶段**表现——projector 学到了 vision→language 分布对齐（生成完整英语 + topic partial matching），但细节欠拟合（颜色/性别/动作幻觉）。因训练量 100x 小于 LLaVA v1.5 官方 pretrain (558K pairs × 1 epoch)。

**Scale up 路径**：完整 5 epochs (预估 loss 收敛至 2.0-2.3) · 或扩到 50K+ pairs · 或加 SFT (Instruct) 阶段

### D2.6 · 正式评测 (BLEU-4 + Baseline 对比)

数据划分：Flickr1K 1000 图 按图划 **900 train / 100 val**（避免 caption-level leakage）
评测：100 张 val 图, greedy decoding (temperature=0.0), max 30 new tokens/图
指标：**手写 BLEU-4** (multi-reference, Chen & Cherry method-1 smoothing, 免第三方依赖)

| 模式 | 生成方式 | **BLEU-4** | 平均生成长度 | 与图关联 |
|------|---------|-----------|-------------|---------|
| **Baseline (未训 projector, 随机初始化)** | 从 100 张 val 图各生成 1 条 | **3.15 %** | 25.7 tokens (发散幻觉) | ❌ 完全无关 |
| **Trained (Flickr1K 300 步)** | 同上 | **17.15 %** | 10.6 tokens (与 GT 13.5 接近) | ✅ topic partial matching |
| **提升倍数** | | **+5.4 ×** | -60% length | 从"编造"到"相关" |

**同图对比样例** (baseline vs trained on 同一 val 图):

| 图片内容 | Baseline (未训) | Trained (300 步) |
|---------|----------------|------------------|
| 蓝白足球员空中跳 | "a person holding a book...reading a book" ❌ | "Two girls are playing a game of soccer." ✅ |
| 男孩坐楼梯滑板 | "a stylized landscape with rolling green hills" ❌ | "A man is sitting on a bench." ✅ |
| 女跑者+观众 | "a futuristic cityscape with towering skyscrapers" ❌ | "A woman in a red dress is standing on a street corner..." ✅ |

**结论**：3.5M 参数的 projector 单独训练 (300 步 · 4500 pairs · 0.5 epoch) 已把 vision→language 分布对齐做到"可辨识 topic"水位。细节精度需要 scale up 到完整 epoch 或加入 SFT 阶段。

### D2.7 · Jetson 迁移分析

参见 [`benchmark/migration_analysis.md`](benchmark/migration_analysis.md)。核心结论：

- **4060 → Orin NX/Nano** (Ampere): bf16 权重零转换, 代码 100% 兼容, 推理速度按内存带宽比例 (272→102 GB/s ≈ 40%)
- **4060 → Xavier NX** (Volta, sm_72): 需 bf16→fp16 权重转换 + 精度验证 (预期 <1% BLEU 掉)
- **GGUF Q4 量化** 跨平台一致 (`.gguf` 硬件无关)
- **诚实标注**: 所有非 4060 数据均为预估, 需真机 rerun 验证

## 架构 (LLaVA v1.5 风格)

```
Image → CLIP-ViT-L/14@336 (冻结) → [B, 576, 1024] visual features
                                ↓
                         2 层 MLP Projector (唯一可训层, 4.20M 参数)
                                ↓
                         [B, 576, 1024] projected tokens
                                ↓
Text prompt (含 <image> 占位符) ─┬──→ 拼装为 [text_pre] [visual] [text_post]
                                ↓
      Qwen3-0.6B (冻结; stage-2 时 q/v 加 LoRA) → text output
```

## 环境 (当前 · Ubuntu / RTX 5060 Ti)

conda env `dl` (Python 3.11, **torch 2.11+cu128**, RTX 5060 Ti 16G, Blackwell sm_120)。
Blackwell 需 torch ≥ 2.7 / CUDA 12.6+;依赖:

```bash
conda activate dl
pip install transformers accelerate sentencepiece pillow huggingface_hub
```

> 前期 4060 平台用 conda env `cu12` (torch 2.6.0+cu124, Windows)。同一份代码零改动跨两平台运行。

### 快速开始

```bash
# 前向验证 (参数量 / shape / VRAM)
python tests/test_forward.py

# stage-1 · projector 对齐 (CLIP-L/14@336 + Qwen3-0.6B, Flickr8k)
python train.py --data data/flickr8k/train.jsonl \
                --val-data data/flickr8k/val.jsonl \
                --image-root data/flickr8k/images \
                --vision openai/clip-vit-large-patch14-336 \
                --llm models/Qwen3-0.6B \
                --steps 3000 --batch 4 --grad-accum 4 --lr 2e-4 \
                --dtype bf16 --val-every 50 \
                --out checkpoints/projector_stage1_qwen3.pt

# stage-2 · LoRA(q/v)+ projector 联合 SFT (LLaVA-Instruct + 平衡 VQAv2)
python train_sft.py --data data/llava_instruct/sft_mix2.json \
                --image-root data/coco/train2014 \
                --vision openai/clip-vit-large-patch14-336 --llm models/Qwen3-0.6B \
                --init-projector checkpoints/projector_stage1_qwen3_best.pt \
                --steps 700 --batch 1 --grad-accum 16 --lr 2e-4 --lora-rank 16 \
                --out checkpoints/vlm_stage2_mix2

# 评测: captioning (corpus+sentence BLEU) · POPE 幻觉
python evaluate.py --ckpt checkpoints/projector_stage1_qwen3_best.pt \
                   --data data/flickr8k/test.jsonl --image-root data/flickr8k/images --max-samples 1000
python benchmark/evaluate_pope.py --projector-ckpt checkpoints/vlm_stage2_mix2/projector.pt \
                   --lora-adapter checkpoints/vlm_stage2_mix2/lora_adapter \
                   --pope-dir data/pope --image-root data/coco/val2014

# 端侧多模态推理 (llama.cpp mmproj, 见 docs/llamacpp_pipeline.md)
llama-mtmd-cli -m models/gguf/qwen3-stage2-merged-q4_k_m.gguf \
               --mmproj models/gguf/mmproj-model-f16.gguf --image <img>.jpg -p "Describe this image."
```

**模型首次需从 HuggingFace 下载**(之后 `~/.cache/huggingface/` 复用):
CLIP-ViT-L/14@336 (~1.7 GB) · Qwen3-0.6B (~1.2 GB)。CLIP-B/32 & SmolLM2-360M / Qwen2.5 为早期归档轨。

## 目录结构

```
vlm/
├── README.md             # 本文件
├── requirements.txt      # 依赖清单
├── setup_env.md          # 环境详情
├── .gitignore
├── model/
│   ├── __init__.py
│   ├── vision_encoder.py # CLIP-ViT 视觉塔 (冻结, 分辨率/维度按加载的模型自适应)
│   ├── projector.py      # 2 层 MLP (唯一可训层, 维度从 vision/llm config 自动对齐)
│   └── vlm.py            # 三段拼装 + forward + generate (ChatML eos 含 <|im_end|>)
├── tests/
│   └── test_forward.py   # 前向验证
├── data/                 # Flickr1K 真数据 + toy 合成数据 (.gitignore)
├── checkpoints/          # 权重存档 (.gitignore)
├── train.py              # projector 训练 (支持 --grad-accum / --init-projector 微调)
├── evaluate.py           # corpus + sentence BLEU-4 评测
└── inference.py          # 单图推理
```

## 关键决策 (面试可讲)

见规划文档第七章 "面试 talk shop 备料"。核心 5 问：

1. **为什么 2 层 MLP 不用 Q-Former?** LLaVA v1.5 消融证明 MLP 优于 Q-Former, 参数少 10x, 训练成本低 10x
2. **为什么冻结 CLIP + LLM?** CLIP 已对齐语言语义, projector 只学线性映射; LLM 冻结保通用能力
3. **端侧部署瓶颈?** KV-cache 内存主导 + CLIP 一次性 encode 首 token 慢
4. **量化后精度如何评估?** PPL 数值 + VQA 定性 20 张对比
5. **如何迁移到 DJI 机载?** 硬件层 NPU+DSP 拆分, 场景层航拍 prompt, 数据层航拍数据集

## 变更记录

- 2026-07-11 v0.1: skeleton 初始版本 (4060, CLIP-B/32 + SmolLM2-360M)
- 2026-07-13 v0.2: 迁移 Ubuntu/RTX 5060 Ti; 架构升级 CLIP-L/14@336 + Qwen2.5-0.5B-Instruct;
  corpus BLEU-4 20.59% (旧 SOTA ~17%); 修跨平台 ckpt 路径 + ChatML eos; Orin NX 兼容性审计通过
- 2026-07-14 v0.3: Flickr8k proper split 干净结果 (BLEU-4 31.73 / CIDEr 92.7); torchao int8/int4 混合精度量化
- 2026-07-14 v0.4 (**selfspec 对齐**): LLM 换 Qwen3-0.6B; 两阶段训练 (projector 对齐 → LoRA SFT);
  POPE 幻觉评测 avg F1 78.59 + 数据配比消融链; llama.cpp GGUF Q4_K_M (0.48GB/3.12×) + llama-server SSE 流式 + fp16↔Q4 PPL (19.63→21.35)
- 2026-07-15 v0.5 (**旗舰统一到 Qwen3 + 归档**): Flickr8k captioning 旗舰迁到 Qwen3 (BLEU-4 32.91 / CIDEr 94.0, 略胜旧 Qwen2.5 轨);
  mmproj 多模态集成端到端跑通; Qwen2.5 / SmolLM2 等未对齐 spec 产物归档至 `checkpoints/_archive_non_spec/`
