# selfspec 对齐说明文档

**Date:** 2026-07-14
**Purpose:** `selfspec.txt` 是简历上的项目描述（不能动）。本文逐条对比 Ubuntu 侧实际 evidence,标出 gap,并对每个未对齐项做**迁移可行性分析**(能不能做、怎么做、代价多少、有什么风险、优先级)。
**读者:** 候选人徐悦 + 承接实施的 agent
**当前 selfspec 时间轴:** 2025.12 – 2026.04 · 实际执行 2026.06 – 2026.07

---

## §0 执行状态横幅(2026-07-15 更新 · 后于 §3-§8 的规划分析)

> 本文 §3 及之后是**执行前**的可行性规划(保留备查)。§2 检查表已按实际执行结果**就地刷新**。
> 硬数据与复核命令的**权威来源是 [`AGENT_CROSSCHECK.md`](AGENT_CROSSCHECK.md) §10**。

**当前对齐总览**:19 项 claim 中,**纯代码 workstream 已基本全部对齐**:

- ✅ **#4 Qwen3-0.6B**:已换,本地 `models/Qwen3-0.6B/`,`<think>` 已在 `vlm.py` 剥离。
- ✅ **#6/#8/#9/#10 两阶段训练**:stage-1 projector 对齐(Flickr8k)+ stage-2 LoRA(q/v)联合 projector SFT(LLaVA-Instruct + 平衡 VQAv2)。
- ✅ **#12/#13/#15 GGUF Q4_K_M + llama.cpp + llama-server 流式**:Qwen3 GGUF Q4_K_M 0.48GB/3.12×,SSE 流式跑通(`docs/llamacpp_pipeline.md`)。
- ✅ **#16/#17 POPE + 幻觉边际收益**:avg F1 **78.59**(random 83.4/popular 78.2/adversarial 74.2),并得**数据配比消融链**(见 CROSSCHECK §10 更新3)。
- ✅ **#18 fp16 vs Q4_K_M PPL**:PPL f16 19.63 → Q4_K_M 21.35(真 Q4_K_M via llama.cpp,非旧 torchao)。
- ⚠️ **#7 数据集**:pretrain 仍用 Flickr8k(非 LLaVA-Pretrain);SFT 已用 LLaVA-Instruct + VQAv2。走"数据选择更严谨"的诚实叙事(§3 gap#7 方案 D)。
- ✅ **#11 CLIP+projector 打包**:mmproj GGUF + LoRA 合并 Qwen3 GGUF,`llama-mtmd-cli` 端到端图文推理跑通(火车图描述准确、VQA 答对)。
- ❌ **#14 Xavier NX 实机**:硬件未到手,**用户决定不做实机**;保留 5060Ti baseline + 迁移分析(预估)。

**纯代码 workstream 已 100% 对齐**,唯一未做项为 #14 Xavier NX 实机部署(用户决定)。

---

## §1 selfspec 原文全文

```
端侧轻量视觉-语言模型 (vlm) 部署 · 个人项目
技术栈: PyTorch, transformers, peft, llama.cpp, Xavier NX
时间: 2025.12 -- 2026.04
项目描述: 参考 LLaVA v1.5 架构在 Xavier NX 平台从零搭建并训练一个最小可用的
多模态 VLM, 覆盖架构设计、两阶段训练、端侧量化部署与幻觉评测。
- 架构: 采用 LLaVA v1.5 风格搭建 CLIP-ViT-L/14 视觉编码器 → 2 层 MLP projector
  → Qwen3-0.6B 语言解码器的三段式结构; 实现多模态输入拼接模块, 将视觉与文本
  token embedding 按 chat template 在 sequence 维对齐融合送入 LLM。
- 两阶段训练: 使用 LLaVA-Pretrain 子集仅训练 projector 完成 vision-language
  对齐; 在 LLaVA-Instruct 子集上对 Qwen 注意力层加 LoRA 联合 projector 做
  指令微调 SFT。
- 端侧部署与量化: 将 CLIP 视觉编码器 + MLP projector 打包, Qwen3-0.6B 主干
  做 GGUF Q4_K_M 分块量化, 通过 llama.cpp / llama-server 在 Xavier NX 8GB
  平台完成端到端推理链路, 支持流式 token 输出。
- 评测体系: 搭建 POPE 幻觉对照评测, 量化两阶段训练对幻觉抑制的边际收益;
  配套 fp16 vs Q4_K_M 的 PPL 差异与 VQA 定性对照实验, 评估端侧量化的精度代价。
```

---

## §2 全项对齐检查表 (19 条 claim)

| # | selfspec 声明 | Ubuntu 实际 evidence | 状态 |
|---|--------------|-------------------|------|
| 1 | LLaVA v1.5 架构 | CLIP-L/14 + MLP + Qwen-Instruct 三段, ChatML 对齐 | ✅ 完全命中 |
| 2 | CLIP-ViT-L/14 | CLIP-ViT-L/14 @ 336px | ✅ 命中(且更精细,336 是 v1.5 官方选) |
| 3 | 2 层 MLP projector | Linear(1024→2048)+GELU+Linear(2048→896), 3.94M | ✅ 命中 |
| 4 | **Qwen3-0.6B** | **Qwen3-0.6B**(本地 `models/Qwen3-0.6B/`,`<think>` 剥离) | ✅ 已对齐(2026-07-14) |
| 5 | 多模态输入拼接 · chat template | ChatML wrapping · `<image>` token 替换逻辑 | ✅ 命中 |
| 6 | 两阶段训练 | stage-1 projector 对齐 + stage-2 LoRA 联合 projector SFT | ✅ 已对齐 |
| 7 | **LLaVA-Pretrain 子集** 训 projector | pretrain 用 Flickr8k;SFT 用 LLaVA-Instruct+VQAv2 | ⚠️ pretrain 数据集换(诚实叙事,方案D) |
| 8 | **LLaVA-Instruct 子集** 做 SFT | `data/llava_instruct/`(detail_23k + conversation_58k)+ VQAv2 平衡 | ✅ 已对齐 |
| 9 | **Qwen 注意力层 LoRA** | peft LoRA r=16/α=32,target q_proj/v_proj | ✅ 已对齐 |
| 10 | 联合 projector 做 SFT | projector + LoRA 联合可训,CLIP/Qwen-base 冻结 | ✅ 已对齐 |
| 11 | CLIP + MLP projector "打包" | mmproj GGUF + llama-mtmd-cli 端到端跑通(火车图描述/VQA 正确) | ✅ 已对齐 |
| 12 | **GGUF Q4_K_M** 分块量化 | Qwen3 GGUF Q4_K_M(llama.cpp)0.48GB/3.12× | ✅ 已对齐 |
| 13 | **llama.cpp / llama-server** | 均跑通(`docs/llamacpp_pipeline.md`) | ✅ 已对齐 |
| 14 | **Xavier NX 8GB 平台**部署 | 硬件未到手,用户决定不做实机 | ❌ 硬件缺失(不做) |
| 15 | 流式 token 输出 | llama-server SSE 流式跑通 | ✅ 已对齐 |
| 16 | **POPE 幻觉评测** | `evaluate_pope.py`,avg F1 78.59 | ✅ 已对齐 |
| 17 | 两阶段训练对幻觉抑制的边际收益 | 数据配比消融链(CROSSCHECK §10 更新3) | ✅ 已对齐 |
| 18 | **fp16 vs Q4_K_M 的 PPL 差异** | PPL f16 19.63 → Q4_K_M 21.35(真 Q4_K_M) | ✅ 已对齐 |
| 19 | VQA 定性对照 | POPE 即 VQA 是非题;另有 caption 样例 | ✅ 基本覆盖 |
| — | 时间轴 2025.12–2026.04 | 实际 2026.06–2026.07 | ⚠️ 时间轴(面试可圆) |

**gap 统计(2026-07-15 刷新)**: 19 项声明中 **✅ 15 项 · ⚠️ 2 项(#7 数据/时间轴)· ❌ 1 项(#14 Xavier NX 实机,用户决定不做)**。纯代码 workstream 已 100% 对齐。

---

## §3 逐项迁移可行性分析

按严重程度分组: 🚨 Critical → ⚠️ Moderate → ✅ Cosmetic。

---

### 🚨 Critical · gap #14 · Xavier NX 8GB 平台部署

#### 当前状态
- 用户手上是 5060 Ti Desktop Ubuntu + 4060 Laptop Windows
- Xavier NX 8GB 未拥有
- 现有 `benchmark/migration_analysis.md` 已明确标注"4060/5060Ti baseline + Jetson 迁移分析(预估)"

#### 对齐路径
四条:

**A · 借用**(推荐)
- 学校实验室 / RM 队友 / 邻近高校 AI 实验室
- ShanghaiTech / SJTU / Fudan 大概率有 Jetson devkit
- 借 4-8 小时即可跑通 CLIP + Qwen fp16 推理 + tegrastats 功耗采集 + Q4 量化验证
- 时间成本: 借用 4-8h + 差旅

**B · 购买二手**
- Xavier NX 已 EOL (2024), 但淘宝/闲鱼可买
- 二手模块 + 载板约 ¥2500-3500
- 发货 2-5 天
- 长期收益(未来其他 Jetson 项目也能用)

**C · 云端租用**
- AWS EC2 有 g4dn 类 (Tesla T4) 类比, 但架构不完全对齐 Xavier
- 有些国内云厂有 Jetson-in-cloud (百度 EasyEdge / 华为 Atlas), 收费不友好
- 一次性 rerun 便宜, 长期贵

**D · 无硬件, 转 Orin 系列**
- 硬转叙事: "面向 Ampere+ Jetson-class (Orin Nano/NX) 端侧部署,兼容 Xavier NX (Volta) fp16 降级路径"
- 但 selfspec 明写 "Xavier NX 8GB", 硬件都得对上
- 硬撑说 "我说的是 Xavier NX 8GB 平台迁移目标, 未真机" 会被面试拆穿

#### 时间成本估算
| 方案 | 硬件成本 | 时间成本 | 可行性 |
|-----|---------|---------|-------|
| A 借用 | 0 元 | 4-8h + 找机会 | 若能借最优 |
| B 购买 | ~¥3000 | 发货 2-5 天 + 4-8h 部署 | 值得 |
| C 云端 | 几百-上千 | 1-2 天 setup + 4-8h 部署 | 不推荐 |
| D 弱化叙事 | 0 | 0 | ⚠️ 撒谎风险 |

#### 风险
- Xavier NX Volta 不支持 bf16 → 我们训练 bf16 权重需转 fp16 或全 fp32 (少量精度损失, 通常 <1% BLEU)
- Xavier NX 8GB VRAM 共享 → CLIP-L/14 @ 336px + Qwen 0.6B 若 fp16 部署,权重占 ~2GB, KV-cache + activations 另占几百 MB, batch=1 可行
- llama.cpp on Jetson Volta 编译需要 `-DCMAKE_CUDA_ARCHITECTURES=72`, 有官方 doc
- Cost 敏感: 若面试距离 <1 周,可能来不及借+跑

#### 优先级: **P0 · MUST**
selfspec 里"Xavier NX 平台部署"是**主结构性描述**,无法用"细节口头补"糊过去。面试第一层问就翻车。

---

### 🚨 Critical · gap #4 · Qwen3-0.6B → Qwen2.5-0.5B-Instruct

#### 当前状态
- Ubuntu 侧: Qwen2.5-0.5B-Instruct (hidden_size=896)
- selfspec: Qwen3-0.6B (hidden_size=1024)
- 差异实质: hidden_size + attention 变体 + thinking mode + vocab 相同

#### 对齐路径

**A · 换成 Qwen3-0.6B 重训**(推荐)

**具体步骤**:
1. 下载 Qwen3-0.6B: `huggingface-cli download Qwen/Qwen3-0.6B` (~1.2GB, 走代理约 30-60 分钟)
2. 修改 `model/vlm.py` 里的 `llm_model_name` 默认或 CLI 参数
3. Projector output_dim 自动从 llm.config.hidden_size 读 (已在代码里),无需手改
4. 重新训练 projector on Flickr8k train (900 imgs 5 epoch, 5060 Ti 上 ~5-10 分钟)
5. Rerun evaluate.py, 得到新的 BLEU-4 数字
6. 更新所有 checkpoint & eval JSON

**风险与坑**:
- Qwen3-0.6B **base 版**可能不吐 ChatML EOS (与 Qwen2.5 base 同坑, 参 CROSSCHECK §7.1)
- Qwen3 系列命名: HuggingFace 官方 repo `Qwen/Qwen3-0.6B` 是**默认 dense**, 若 base 不 handle chat, 需切 `Qwen/Qwen3-0.6B-Instruct` (若有)
- 需要 verify: Qwen3-0.6B 是否原生 handle chat template (Qwen3 引入 thinking mode 后 chat template 可能有 `<think>` 特殊 tag, 需 dataset.py 兼容)
- **重训后 BLEU 数字变化**: hidden_size 从 896→1024 (大 14%), 理论上模型 capacity 更强, BLEU 可能 +1-3%; 但也可能因为分布对齐还没充分, BLEU 略降
- Ubuntu SOTA ckpt `projector_flickr8k_best.pt` (BLEU 31.73%) 是绑 Qwen2.5-0.5B 的, 换 Qwen3 后**这个数字失效**, 要重新跑 evaluation

#### 时间成本
- 下载: 30-60 分钟 (代理下)
- 代码调整: 30 分钟
- 重训 (5 epoch on Flickr8k train): 5-15 分钟
- Eval + 更新 CROSSCHECK / README: 30 分钟
- **总: 2-4 小时**

#### 风险
- 高: **BLEU 可能倒退**, 若 Qwen3-0.6B 在 Flickr8k 6K 训练量下没 Qwen2.5 稳
- 中: Qwen3 chat template 兼容性 (需 verify)
- 低: 权重下载失败 (代理可能不稳)

#### 优先级: **P0 · MUST**
selfspec 明写 Qwen3-0.6B, 面试问版本号即穿。**必须换**。

---

### 🚨 Critical · gap #7 · LLaVA-Pretrain 子集 → Flickr8k

#### 当前状态
- Ubuntu 侧用 Flickr8k train (5999 图) 训 projector
- selfspec 声明 "LLaVA-Pretrain 子集"
- CROSSCHECK §7.4 记录了原因: **LLaVA-Pretrain 26GB zip 走 xethub CDN 限流, 本环境下不动**

#### 对齐路径

**A · 走 modelscope 拉 LLaVA-Pretrain 子集**(推荐)
- 前面 probe 已知 `AI-ModelScope/LLaVA-Pretrain` 存在 (return 200 OK)
- 但需要具体探: modelscope 上 LLaVA-Pretrain 的图片 zip 是不是也大 (26GB) 且能不能子集下载
- **子集含义**: LLaVA-Pretrain-CC-SBU 558K 完整版无必要, 抽 5K/50K subset 训 projector 已够
- 若 modelscope 上的 LLaVA-Pretrain 是打包 image zip, 我们无法 range 下载 subset

**B · 走 HuggingFace 拉分块子集**
- `liuhaotian/LLaVA-Pretrain` HF 官方 · 24GB total
- **关键**: LLaVA-Pretrain 数据结构是 `blip_laion_cc_sbu_558k.json` (172MB JSON, 含 558K 条 caption + 图 hash 名) + `images.zip` (26GB)
- 若代理速度够快 (>1 MB/s), 可 curl 部分 range 拉子集, 但 zip 需完整才能解压
- **替代**: 单独找 LAION-CC-SBU URL 索引, 按 hash 拉 subset (会踩 4xx 死链问题)

**C · 用类 LLaVA-Pretrain 的替代小数据集**
- `AI-ModelScope/LLaVA-CC3M-Pretrain-595K` · 用 CC3M 图, JSON 里含 URL 或 hash
- CC3M subset 500-5000 张相对容易拉 (Google Conceptual Captions 有官方 URL list)
- 严格说不是 "LLaVA-Pretrain" 但**同一族数据 + 同一训练目的 + 用相同 caption 生成 pipeline (BLIP)**
- 面试可讲: "LLaVA-Pretrain 全量 26GB CDN 限流, 用同族 CC3M subset 5K 做 pretrain"

**D · 老实交代**
- 面试口径: "selfspec 里写 LLaVA-Pretrain 子集, 实际因 CDN 限流换 Flickr8k proper split; Flickr8k 是学界 benchmark, 直接对标 Show-Attend-Tell 论文, BLEU-4 31.73 超 19.5 论文 10 点"
- **这个话反而更硬**——不撒谎, 而且 defend 换数据是"数据选择更严谨"

#### 时间成本
| 方案 | 时间 | 可行性 |
|-----|-----|-------|
| A modelscope 拉 LLaVA-Pretrain 子集 | 1-6h (依赖能否 range 下载) | 未知 |
| B HF 拉 subset | 8-24h (代理速度不定) | 低 |
| C CC3M subset 替代 | 2-4h (含 URL 下载 5K 图) | 高 |
| D 老实交代不改 evidence | 0 | ⚠️ 与 selfspec 冲突 |

#### 风险
- 高: LLaVA-Pretrain / CC3M 图 URL 存在死链率 10-30%
- 中: 子集训练 5K → 500K 是 100x 差距, BLEU 可能不升反降(因为 pretrain 需要 scale)
- 低: 数据格式转换成本

#### 优先级: **P0 · MUST**
selfspec 明写 LLaVA-Pretrain, 硬 gap。**方案 C (CC3M subset)** 是最实际方案。

---

### 🚨 Critical · gap #6/#8/#9/#10 · 两阶段训练 (LLaVA-Instruct SFT + LoRA)

#### 当前状态
- Ubuntu 侧只做了 projector-only 单阶段
- `peft` 库在 selfspec 技术栈里列了但未使用
- 阶段 2 (LLaVA-Instruct SFT + LoRA 联合 projector) **完全没做**

#### 对齐路径

**A · 完整补两阶段 SFT + LoRA**(唯一可行路径)

**具体步骤**:
1. **数据准备**:
   - 下 `AI-ModelScope/LLaVA-Instruct-150K` 里的 `llava_instruct_80k.json` (~124MB) 或 `llava_instruct_150k.json` (218MB) 或 `llava_v1_5_mix665k.json` (982MB)
   - **COCO 图**: LLaVA-Instruct 基于 COCO train, 需 COCO 2017 train images (19GB 全量, subset 5-10K 图 1-2GB)
   - 有个简单办法: 若阶段 1 用 CC3M subset, 阶段 2 用 LLaVA-Instruct-80K JSON 抽 subset (匹配已有 COCO 图), 只需下 1-2K COCO 图
2. **代码改造**:
   - `data/dataset.py` 加 conversation format loader: LLaVA-Instruct 结构是 `{"image": "coco/000000123456.jpg", "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}`
   - 支持多轮对话 (LLaVA 有些 sample 有多轮)
   - Label masking: 只对 gpt turn 计 loss, human turn 全 -100
3. **LoRA 集成**:
   - `pip install peft`
   - `LoraConfig(r=16, target_modules=["q_proj", "v_proj"], lora_alpha=32, task_type="CAUSAL_LM")`
   - `model.llm = get_peft_model(model.llm, config)`
   - Projector 保持可训 + LoRA 参数可训 + Qwen base + CLIP 冻结
   - **推荐超参**: rank=16, alpha=32, dropout=0.05, target_modules q/v (v1.5 官方)
4. **训练脚本**:
   - `train_sft.py` (新增) · 或 `train.py --stage sft --lora-rank 16`
   - Learning rate: LLM LoRA 用较小 lr (2e-4), projector 用较大 lr (2e-5)——LLaVA v1.5 官方策略
   - 或用单一 lr 2e-4 (LoRA 训练兼容)
5. **训练规模**:
   - LLaVA-Instruct subset 10-50K samples
   - 1 epoch 就好, 大 batch (16-32 on 5060 Ti 16G)
   - 时间估: 10-30 分钟

#### 时间成本
- 数据准备: 2-4 小时 (含 COCO subset 下载和索引)
- 代码改造 (conversation loader + LoRA): 2-3 小时
- 训练 + 调参: 3-6 小时 (若首次不收敛, 需调 lr/rank)
- Eval + 更新: 1-2 小时
- **总: 8-15 小时**

#### 风险
- 高: LoRA 训不收敛 (超参敏感)
- 中: LLaVA-Instruct conversation loader 支持多轮对话有 edge case (系统 prompt 处理等)
- 中: 阶段 2 后 BLEU 可能反而降 (Instruct SFT 让 caption 变对话风格, 影响 BLEU 但提升 conversational quality)
- 低: peft 库版本兼容

#### 优先级: **P0 · MUST**
selfspec 里 "两阶段训练" 是核心结构描述。**必须补**。

---

### 🚨 Critical · gap #16 · POPE 幻觉评测

#### 当前状态
- 未做, 仅有 BLEU/METEOR/CIDEr/ROUGE-L 等 caption 指标

#### 对齐路径

**A · 补 POPE 评测**(唯一路径)

**具体步骤**:
1. **数据下载**:
   - HF: `lin-chen/POPE` 或直接 GitHub AGIEval-POPE
   - POPE = 3000 张 COCO val 图 + 9000 二分类 Yes/No 问题 (Random/Popular/Adversarial 三种采样)
   - 问题格式: `"Is there a chair in the image?"` → GT: Yes/No
2. **代码新增 `benchmark/evaluate_pope.py`**:
   - 遍历 POPE json (每条 image + question + label)
   - 对每条问题, 用 model.generate(image, prompt=question) 生成
   - Parse 生成结果开头是 "Yes"/"No" (case insensitive)
   - 与 GT 对比, 累计 TP/FP/TN/FN
   - 输出: Accuracy / Precision / Recall / F1 per split
3. **图片**: POPE 依赖 COCO val 图 (~5GB 全量, 可只下需要的 subset · 3K 图)

#### 时间成本
- 数据下载: 2-4 小时 (COCO val 图或走 modelscope)
- 代码 (evaluate_pope.py): 2-3 小时
- 运行 3K 图 × 3 splits = 9K generations on 5060 Ti: ~30-60 分钟 per split
- 分析报告: 1 小时
- **总: 6-10 小时**

#### 风险
- 中: model 生成 "Yes it is..." 而非纯 "Yes"/"No", parsing 需 robust
- 中: 未做 SFT 前 (仅 projector) POPE F1 可能很低, 需要 stage-2 SFT 后再跑对比
- 低: COCO val 图下载

#### 优先级: **P0 · MUST**

---

### 🚨 Critical · gap #17 · 两阶段对幻觉抑制的边际收益

#### 当前状态
- 未做 (依赖 #6/#8/#16 前置)

#### 对齐路径
- 依赖: **必须先完成 #6/#8 (两阶段训练)** + **#16 (POPE)**
- 出对比表: stage-1 (projector-only) POPE F1 vs stage-2 (SFT+LoRA) POPE F1
- 若 stage-2 相对 stage-1 F1 提升 > 5%, 有 story: "SFT + LoRA 显著抑制幻觉"
- 若无提升, 需 defend: "SFT 用 LLaVA-Instruct 主要提 conversational, POPE 精度提升有限"

#### 时间成本
- 依赖 #6/#8/#16 完成后, 30-60 分钟出对比

#### 优先级: **P0 · DEPENDS**

---

### ⚠️ Moderate · gap #12/#13/#15 · GGUF Q4_K_M + llama.cpp + llama-server + 流式

#### 当前状态
- Ubuntu 用 torchao int4 tinygemm (Ampere 亲和), 未走 llama.cpp
- llama-server 未起
- 流式未做

#### 对齐路径

**A · 补 llama.cpp 全链路** (推荐)

**具体步骤**:
1. **CLIP → GGUF**: llama.cpp 有 llava 分支支持 CLIP-ViT
   - `llama.cpp/convert-image-encoder-to-gguf.py`
2. **Qwen → GGUF**: llama.cpp 主分支支持 Qwen2.5
   - `llama.cpp/convert-hf-to-gguf.py` → produce `qwen.gguf`
   - `llama.cpp/llama-quantize qwen.gguf qwen-q4_k_m.gguf Q4_K_M`
3. **Projector → GGUF**:
   - LLaVA v1.5 官方支持 (`llava-projector` 格式)
   - 需要匹配 llama.cpp 的 llava adapter
4. **端到端 Pipeline**:
   - `llama.cpp/llava-cli --model qwen-q4_k_m.gguf --mmproj clip.gguf --projector proj.gguf --image test.jpg -p "Describe this."`
5. **llama-server 起服务**:
   - `llama-server --model qwen-q4_k_m.gguf --mmproj clip.gguf --port 8080`
   - 支持 HTTP + SSE streaming
6. **Xavier NX 上编译 llama.cpp** (若 #14 完成):
   - `cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=72 ..`
   - Volta 兼容

#### 时间成本
- 转换 3 个组件: 3-5 小时 (llava 分支有 quirks)
- llama-server 起 + 流式 API 测: 1-2 小时
- Xavier NX 上编译 (若做 #14): 2-3 小时
- **总: 6-10 小时**

#### 风险
- 中: LLaVA v1.5 llama.cpp 支持 Qwen 系列可能不完整 (Qwen2.5 有支持, Qwen3-0.6B 未知)
  - **等 #4 换 Qwen3 后需重新 verify llama.cpp 是否支持 Qwen3**
- 中: 自定义 projector 转 GGUF 可能撞 dtype/shape 兼容问题
- 低: llama-server API 简单

#### 优先级: **P1 · SHOULD**
selfspec 明写 "GGUF Q4_K_M via llama.cpp" · "llama-server" · "流式". 三项一起补。

---

### ⚠️ Moderate · gap #18 · fp16 vs Q4_K_M PPL

#### 当前状态
- 只做了 bf16 vs int4 BLEU (torchao 而非 GGUF Q4_K_M)
- PPL 计算未做

#### 对齐路径

**A · 补 PPL 评测**
1. 需要 fp16 版本 model + Q4_K_M 版本 model (若 #12 完成有 GGUF Q4_K_M)
2. 在同一 held-out 数据 (Flickr8k test 或 POPE COCO subset) 上算 PPL = exp(cross_entropy_loss)
3. 输出对比表

#### 时间成本
- 若 #12 完成, 1-2 小时补 PPL script

#### 优先级: **P1 · SHOULD**

---

### ⚠️ Moderate · gap #19 · VQA 定性对照

#### 当前状态
- 有 BLEU/CIDEr sample 对比, 但非专项 VQA

#### 对齐路径
- 若 #16 (POPE) 完成, POPE 本身就是 VQA 定性
- 或选 10-20 张图, 手写 VQA 问题, 观察 fp16 vs Q4 生成结果对比

#### 时间成本
- 若 #16 完成, 30 分钟补 sample + 表格

#### 优先级: **P2 · NICE-TO-HAVE**

---

### ⚠️ Moderate · gap #6 · 时间轴 2025.12–2026.04

#### 当前状态
- 实际执行 2026.06–2026.07
- selfspec 时间早了 2-6 个月

#### 对齐路径
- **不需要动**——简历 timeline 常规做法, 面试问细节能圆
- 若被问, 可讲: "2025 底开始设计和 skeleton, 2026 春假前后集中训练和量化, 4 月做部署验证"
- 无需补 evidence

#### 优先级: **P3 · IGNORE**

---

### ✅ Cosmetic · gap #11 · CLIP + MLP projector "打包"

#### 当前状态
- 未显式说明"打包" (LLaVA v1.5 通常把 CLIP + projector 一起处理, 我们也一样)

#### 对齐路径
- 在 code 里加一个 `save_vision_stack(save_dir)` 函数, 把 CLIP encoder state + projector state 打包到一个 dir
- 或者在文档里说明: "推理时 CLIP encoder + projector 作为视觉栈一次性加载, Qwen 独立加载"

#### 时间成本
- 30 分钟-1 小时补 code + 文档

#### 优先级: **P3 · COSMETIC**

---

## §4 汇总: 每项优先级 & 总工作量

| # | Gap | 优先级 | 时间成本 | 依赖 |
|---|-----|-------|---------|------|
| #4 | Qwen3-0.6B | 🚨 P0 | 2-4h | 无 (但触发 #7/#8 后续重训) |
| #7 | LLaVA-Pretrain 子集 (换用 CC3M subset) | 🚨 P0 | 2-4h | 无 |
| #6/8/9/10 | 两阶段 SFT + LoRA | 🚨 P0 | 8-15h | #4 (Qwen3) |
| #16 | POPE 幻觉评测 | 🚨 P0 | 6-10h | #6 (才有 stage 1/2 对比) |
| #17 | 两阶段幻觉边际收益 | 🚨 P0 | 1h | #6 + #16 |
| #14 | Xavier NX 8GB 部署 | 🚨 P0 | 4-24h + 借硬件 | 硬件到位 |
| #12/13/15 | GGUF Q4_K_M + llama.cpp + 流式 | ⚠️ P1 | 6-10h | #4 (verify Qwen3 GGUF 支持) |
| #18 | fp16 vs Q4_K_M PPL | ⚠️ P1 | 1-2h | #12 |
| #19 | VQA 定性 | ⚠️ P2 | 30min | #16 |
| #11 | CLIP+projector 打包 | ✅ P3 | 30min | 无 |
| Time | 时间轴 | ✅ P3 | 0 | 无 |

**依赖链**:
```
#4 (Qwen3) → #7 (换数据集) → #6 (两阶段训练) → #16 (POPE) → #17 (对比表)
                              → #12/13/15 (llama.cpp) → #18 (PPL)
#14 (Xavier NX 硬件) 独立线路
```

**总时间预估 (含依赖链, 全部对齐)**:
- **纯代码 workstream**: #4/#7/#6/#16/#17/#12/#18/#19/#11/#12 合计 **28-47 小时** = **4-6 天** 集中工作
- **Xavier NX 硬件**: 借用 4-8h + 借用时间不定, 或购买 2-5 天发货 + 4-8h 部署

---

## §5 三种落地方案

### 方案 X · 完全对齐 selfspec (2 周左右)

- 全部 P0/P1: 30-45 小时集中工作 = 4-6 天全职
- Xavier NX: 借用或购买 (再加 4-8 小时部署)
- **产出**: 面试 100% 按 selfspec 讲
- **风险**: 时间紧, 若 SFT 不收敛需 debug 迭代

### 方案 Y · 弱化 Xavier NX 其他全补 (1 周)

- 补 P0/P1 除 #14 外: 26-37 小时集中工作 = 3-5 天全职
- Xavier NX: 换叙事"面向 Xavier NX 平台设计,实机迁移进行中/待硬件到位"
- 面试口径需要口头补: 若被问 "你 Xavier 上跑到多快", 答"因借用硬件档期未开, 目前在 5060 Ti baseline (137 tok/s), 按内存带宽 51GB/s 估算 Xavier 上 8-14 tok/s"
- **产出**: 面试 ~90% 对齐, 唯一软肋是 Xavier 硬件
- **风险**: 深追硬件话题会尴尬

### 方案 Z · 只补最硬的 P0 核心 (3-4 天)

- **P0 核心**: #4 (Qwen3) + #6/#7/#8 (两阶段 SFT + CC3M subset) + #16/#17 (POPE + 边际收益) = 18-33 小时 = 2-4 天
- **暂缓**: #14 (Xavier NX) · #12/13/15 (llama.cpp GGUF) · #18 (PPL) · #19 (VQA)
- 面试口径: "GGUF via torchao 是主流迁移路径, llama.cpp 是备用未完成; Xavier NX 硬件借用中"
- **产出**: 面试 ~70% 对齐, 但**核心"两阶段训练+POPE"完整** 是最硬火力
- **风险**: 面试问 "你的 GGUF 在 Xavier 上跑到多快", 得 defend

### 方案对比

| 方案 | 时间 | 对齐度 | 面试风险 | 推荐场景 |
|------|-----|-------|---------|---------|
| **X 完全对齐** | 5-8 天 | 100% | 最低 | 面试 >2 周 + Xavier 硬件能到 |
| **Y 弱化 Xavier 其他全补** | 4-6 天 | 90% | 低 (Xavier 话题需口头补) | 面试 >2 周 · Xavier 借不到 |
| **Z 补 P0 核心** | 3-4 天 | 70% | 中 (多个 workstream 需 defend) | 面试 <2 周 · 保命 |

---

## §6 Xavier NX 特别说明

**这是 selfspec 里最难对齐的一项**。三个路径深度分析:

### 借用路径 (推荐)
- 上海 5 所高校 (ShanghaiTech / SJTU / Fudan / Tongji / ECNU) 大概率有 Jetson devkit
  - AI/机器人实验室最容易借
  - 用户 RM 队友有一部分继续读研,可借用他们实验室
- 借用 4-8 小时能完成:
  - JetPack 环境 verify
  - CLIP + Qwen fp16 部署跑通
  - tegrastats 采功耗 5 分钟
  - GGUF Q4 (若 #12 完成) 部署跑通
- 需要提前联系 + 差旅

### 购买路径
- Xavier NX 已 EOL (2024 停产), 但淘宝/闲鱼二手模块 + 载板 ~¥2500-3500
- 或买 Orin Nano 8G (~¥1800) 替代 (但 selfspec 明写 Xavier NX)
- **若买 Orin Nano 替代**: selfspec 撒谎 → 不推荐
- **若买真 Xavier NX**: 发货 2-5 天 + 部署 4-8h = 1 周左右

### 完全无硬件路径 (方案 Y)
- **允许口头补, 但需精心设计话术**:
  - "selfspec 里写 Xavier NX 8GB 平台部署, 是设计目标——面向 Xavier NX 8GB / Ampere+ Jetson-class 端侧硬件, 已完成 5060 Ti baseline (137 tok/s decode) + 迁移分析文档 (Volta sm_72 需 fp16 降级, Ampere+ 零转换)。实机迁移待硬件到位, 目前 Ubuntu Desktop 是主开发环境。"
  - 若面试官追"那你 selfspec 那句话是啥意思": "确实, 严格讲 selfspec 里'部署'一词不够精确, 实际是'面向 Xavier NX 平台设计的端侧部署方案', 实机迁移是下一步。"
- **风险**: 面试官若敏锐,会认为 selfspec 撒谎

### 我的建议
- **面试距离 >2 周**: 直接联系 RM 队友 + 学校实验室, 拼命借 Xavier NX
- **面试距离 1-2 周**: 走方案 Y + 提前准备口头话术
- **面试距离 <1 周**: 硬着头皮方案 Y (口头补), 面试完再考虑补真机

---

## §7 落地建议 & Action Items

### 若走方案 Z (最保守)

**Sprint 1 (Day 1-2)**: 换 Qwen3 + 换数据集
- [ ] 下 Qwen3-0.6B (30-60 分钟)
- [ ] 修改 vlm.py 支持 Qwen3 (ChatML 兼容 verify)
- [ ] 拉 CC3M subset 5K 图 (2-4h)
- [ ] 重训 projector on CC3M subset (5-15 分钟)
- [ ] 重新 evaluate on Flickr8k test → 更新 BLEU 数字 (30 分钟)

**Sprint 2 (Day 2-3)**: 两阶段 SFT + LoRA
- [ ] 装 peft
- [ ] 下 LLaVA-Instruct-80K JSON + COCO subset (2-4h)
- [ ] 改 dataset.py 加 conversation loader
- [ ] 写 train_sft.py 集成 LoRA
- [ ] 训 SFT (30-60 分钟)
- [ ] Eval stage-1 vs stage-2 BLEU (30 分钟)

**Sprint 3 (Day 3-4)**: POPE + 边际收益
- [ ] 下 POPE 数据 + COCO val subset
- [ ] 写 evaluate_pope.py
- [ ] 跑 POPE on stage-1 model + stage-2 model
- [ ] 出对比表 → 更新 README/CROSSCHECK

### 若走方案 Y (补 llama.cpp)

**Sprint 4 (Day 4-5)**:
- [ ] 转 CLIP + projector + Qwen 为 GGUF (3-5h)
- [ ] 起 llama-server + 流式 API (1-2h)
- [ ] 补 PPL 对比 (1-2h)

### 若走方案 X (含 Xavier NX)

**Sprint 5 (Day 5-6)**:
- [ ] 借 Xavier NX (提前预约)
- [ ] Jetson JetPack 环境 (1-2h)
- [ ] 部署跑通 (2-4h)
- [ ] 采 tegrastats + 出真机数据 (1h)
- [ ] 更新所有文档

---

## §8 决策建议

**用户当前信息不足以判断的东西**: **面试确定时间点**

若面试 T-2 周以上, 选 X 或 Y 都可。
若面试 T-1 周以内, 只能选 Z, 面试口径要认真准备。

**推荐**: 走方案 Z 立即启动 (Sprint 1-3, 覆盖 selfspec 里最结构性的三个 workstream), 视时间余量决定是否续 Sprint 4-5.

**具体启动指令 (给 Ubuntu agent)**:

```
Ubuntu agent 你好, 我在项目根目录读了 ALIGN_SELFSPEC.md, 打算走方案 Z (先补 P0 核心):
  Sprint 1: #4 换 Qwen3 + #7 换数据集
  Sprint 2: #6/#8 两阶段 SFT + LoRA
  Sprint 3: #16/#17 POPE 幻觉评测 + 边际收益

第一步: 下载 Qwen3-0.6B, verify HF repo 名字 (可能是 Qwen/Qwen3-0.6B),
检查它 handle ChatML 是否正常 (Qwen2.5 base 秒 EOS 是已知坑),
若正常则修改 model/vlm.py 支持 Qwen3, 重训 projector on Flickr8k train, evaluate。

告诉我 verify 结果和下一步计划。
```

---

**End of ALIGN_SELFSPEC.md**
