# scratch-vlm · 面试 Talk-shop Q&A

**Date:** 2026-07-13
**用途:** DJI 端侧 AI 系统工程师岗面试备料。按 JD 关键词(深度学习编译器 / 混合精度量化 / 异构调度运行时 / 算子性能调优 / 端侧大模型)+ 项目维度(架构 / 训练 / 评测 / 迁移 / 量化)分组。
**读法:** 每题「先结论、后展开」。数字后括号标注来源:**[实测]** = 本机真跑出的数据;**[预估]** = 基于架构/带宽推算,未真机验证;**[计划]** = 尚未落地的下一步。

**一句话项目定位:** 参考 LLaVA v1.5 架构手工拼装 CLIP + MLP-Projector + LLM 的极简 VLM,**冻结视觉/语言主干、仅训 3.94M 的 projector(占总参 0.49%)**,重点展示端侧 AI 全流程能力(训练 → 推理 → 量化 → 硬件迁移)。当前 SOTA:Flickr1K captioning **corpus BLEU-4 20.59%**,推理显存 **1.6GB**。**[实测]**

---

## 组 A · 架构与设计选型(端侧大模型)

### A1. 用一句话讲清楚你的模型结构和数据流。

**结论:** 三段式 LLaVA v1.5 风格:冻结 CLIP 视觉塔 → 可训 MLP projector → 冻结 LLM。

- 视觉塔:**CLIP-ViT-L/14@336px**(~304M,冻结),输出 `[B, 576, 1024]` 的 patch-level 特征。**[实测]**
- 投影层:**2 层 MLP(1024 → 2048 → 896),GELU 激活,3.94M 参数**,是全模型唯一可训层。**[实测]**
- 语言塔:**Qwen2.5-0.5B-Instruct**(~494M,896 维隐层,冻结)。**[实测]**
- 拼装:prompt 里放一个 `<image>` 占位 token,forward 时把它替换成 576 个投影后的 visual token,与文本 token embedding 拼成 `[text_pre][visual×576][text_post]` 送入 LLM。visual 位置的 label 全置 -100,不算 loss。
- 总参 ~801M,可训占比 **0.49%**。**[实测]**

### A2. 为什么用 2 层 MLP 而不是 Q-Former?

**结论:** LLaVA v1.5 消融已证明 MLP 优于 Q-Former,且对端侧更友好。

- **精度:** LLaVA v1.5 论文消融显示 2 层 MLP 在下游榜单普遍优于或持平 BLIP-2 的 Q-Former。
- **成本:** MLP 参数量约少一个数量级(我们只有 3.94M),训练成本低约 10x,无需 Q-Former 的可学习 query + 交叉注意力那套额外结构。
- **端侧视角:** MLP 是纯 GEMM,任何异构后端(GPU/NPU/DSP)都原生支持;Q-Former 带 attention,算子 fallback 风险更高。
- **本质:** CLIP 已把视觉特征对齐到语言语义空间,projector 只需学一个「维度对齐 + 分布搬运」的近线性映射,不需要 Q-Former 那种重的信息压缩模块。

### A3. 为什么冻结 CLIP 和 LLM,只训 projector?

**结论:** 这是「冻结主干 + 微型 adapter」的端侧经济学——用 0.49% 的可训参数站在两个预训练主干的肩膀上。

- CLIP 已经把图像对齐到语言语义,视觉表征够用,不必重训。
- LLM 冻结可**保住通用语言能力**,避免灾难性遗忘;同时权重可离线一次性转换/量化,天然适配任意异构后端。
- 只训 projector → 训练显存/算力需求极低,单卡 8G 消费级 GPU 就能跑通全流程。
- **经济学证据:** 仅 3.94M 可训参数就把 corpus BLEU-4 从未训的 0.86% 拉到 20.59%(经典 Show-Attend-Tell 在 Flickr30k 约 19-20% 量级),这正是端侧 adapter 范式的性价比体现。**[实测]**

### A4. 视觉特征为什么取 CLIP 倒数第二层、并去掉 CLS token?

**结论:** 沿用 LLaVA v1.5 的经验做法。

- `select_layer = -2`:最后一层特征过度向语言对比目标塌缩、丢视觉细节;倒数第二层保留更多空间/纹理信息。
- `select_feature = "patch"`:去掉 CLS(全局摘要 token),保留 576 个 patch token,给 LLM 更细粒度的空间信息做 captioning。
- 336px ÷ 14 patch = 24×24 = **576 个 patch token**(若用 224px 输入则是 16×16=256)。**[实测]**

---

## 组 B · 训练与「真实踩坑」

### B1.(重点踩坑)base 模型换成 Instruct,BLEU 从 12% 跳回 22%,发生了什么?

**结论:** 冻结 LM 训练不改变其输出分布,base 模型压根不会吐 ChatML 的 `<|im_end|>`,导致生成越过 caption 继续吐垃圾 token,precision 被稀释,BLEU 从 ~22% 掉到 ~12%;换 Instruct 版解决。**[实测]**

- 评测 prompt 用的是 ChatML 格式:`<|im_start|>user\n<image>\nDescribe this image.<|im_end|>\n<|im_start|>assistant\n`。
- **根因:** 我只训 projector、LLM 全程冻结,所以 LLM 的 token 分布不会因训练而学会「说完 caption 就停」。Qwen2.5-0.5B **base** 版的默认 eos 只有 `<|endoftext|>`,不含 `<|im_end|>`——它没在 ChatML 上对齐过,生成完 caption 不会终止,继续输出无关 token。
- **后果:** BLEU 的 n-gram precision 被这些垃圾 token 稀释,分数腰斩(~22% → ~12%),同时平均生成长度发散。
- **两道修复:**(1)换 **Instruct** 版,它原生在 ChatML 上对齐,会自然吐 `<|im_end|>`;(2)代码里 `generate` 显式把 `<|im_end|>` 的 id 也加进 `eos_token_id` 停止集合,双保险。
- **系统工程 takeaway:** 冻结主干的 VLM,**生成终止行为完全继承自底座模型的对齐格式**,选底座和对齐生成模板与训练本身同等重要。

### B2.(重点踩坑)CLIP-L@336 的 576 token,为什么让训练显存暴涨到 batch8 连 16G 都 OOM,推理却只要 1.6GB?

**结论:** 训练要为反向传播保留整条前向路径(含冻结 LLM)的激活,序列长(576 visual + 文本)× batch × 层数把激活撑爆;推理无反向、无激活留存,加上 KV-cache 增量很小,所以只 1.6GB。**[实测:9.9GB 训练 / 1.6GB 推理]**

- **关键点常被误解:** LLM 虽冻结(`requires_grad=False`),但梯度必须**穿过** LLM 反传才能到达 projector。因此每一层的前向激活都要留着做链式求导——激活显存 ≈ O(batch × seq_len × layers × hidden),576 的长序列让它线性放大。
- 所以训练用 **batch=4 × grad-accum=4**(有效 batch 16)来在 16G 内凑大有效 batch;直接 batch=8 会 OOM。训练峰值 **~9.9GB**。**[实测]**
- 推理 batch=1、`torch.no_grad`,不留反向激活;576 个 visual token 只 encode 一次,自回归阶段 KV-cache 增量很小。峰值锁在 **1.6GB**,稳进 8GB,连 Orin Nano 8G 都容得下。**[实测]**
- 对比参照:早期 CLIP-B/32(49 token)时代训练峰值才 1.45~2.67GB。**[实测]** 换 L/14@336 后 token 数 ×~10,显存压力主要来自这里。

### B3. 你的训练配方和收敛情况?

**结论:** AdamW + cosine schedule,只更新 projector,真数据上 loss 稳定下降、val BLEU 显著优于 baseline。

- 当前 SOTA 配方:CLIP-L/14@336 + Qwen2.5-0.5B-Instruct,`--steps 400 --batch 4 --grad-accum 4 --lr 2e-4 --dtype bf16`,从 base 版 ckpt 微调。
- 优化器 AdamW,LR 手动 linear warmup + cosine decay(1.0 → 0.1),梯度裁剪 max_norm=1.0。
- 数据:`nlphuji/flickr_1k_test`,1K 图 ×5 caption,**按图切 900 train / 100 val**(避免同图 caption 泄漏到 val)。
- 早期 4060 基线(CLIP-B/32 + SmolLM2)硬数据:loss 3.57 → 0.82(-77%),197 ms/step,VRAM 1.45GB。**[实测]** 用来佐证 pipeline 正确性。

### B4. 为什么用 bf16 而不是 fp16 训练?

**结论:** bf16 数值更稳且 Ampere/Ada 原生;唯一代价是若目标平台是 Volta(Xavier),需要一步 bf16→fp16 权重转换。

- bf16 有 8-bit 指数,动态范围与 fp32 相同;fp16 只有 5-bit 指数,小 loss 值时易 underflow。
- 4060(Ada)、5060 Ti(Blackwell)、Orin(Ampere sm_87)都**原生**支持 bf16 tensor core,零性能损失、跨平台零权重转换。
- 只有 Jetson Xavier NX(Volta sm_72)不支持 bf16,需 `model.to(torch.float16)`,预期精度掉 <1% BLEU,**但需真机 verify**。**[预估]**

---

## 组 C · 评测方法学(诚信叙事)

### C1. 你的核心结果是什么?怎么证明训练真的有效?

**结论:** Flickr1K 100 张 val,corpus BLEU-4 **20.59%**、sentence(平滑)**22.18%**;未训 baseline corpus 仅 **0.86%**,提升约 **24×**。**[实测]**

| 指标 | Baseline(未训 projector) | Trained | 提升 |
|------|--------------------------|---------|------|
| corpus BLEU-4(可对标) | 0.86% | **20.59%** | +24× |
| sentence BLEU-4(平滑,内部追踪) | 2.99% | 22.18% | +7.4× |
| 平均生成长度 | 24.0 tok(发散) | 11.4 tok(贴近 GT 13.5) | 收敛 |

- Baseline 用**随机初始化的 projector**跑同一套流程,输出与图完全无关(纯幻觉);trained 后能做到 topic 级对齐。这个对照直接证明「是 projector 学到了 vision→language 对齐」,而非底座模型本身会 caption。**[实测]**

### C2. corpus BLEU 和 sentence BLEU 有什么区别,为什么两个都报?

**结论:** corpus 是论文通用报法、可跨论文对标;sentence 平滑版对短句更宽容、数值偏高,只作内部进度追踪。两个都报是为透明诚信。

- **corpus BLEU-4:** 在整个语料上累计 n-gram 命中/总数、按语料总长施加 brevity penalty,无平滑。这是 Show-and-Tell / Show-Attend-Tell 等 captioning 论文的标准报法。
- **sentence BLEU-4:** 逐句算再平均,且用 Chen & Cherry method-1 smoothing 处理 zero-count n-gram。对短句宽容,数值系统性偏高,不能直接和论文比。
- 两者都是**手写实现、免第三方依赖**(modified n-gram precision 1..4 + brevity penalty + 平滑),评测时同屏打印,主动暴露差异。

### C3. 20% 的 BLEU 算好吗?为什么不直接和 COCO SOTA 比?

**结论:** 只作「同量级」参考、非 leaderboard;数据集规模差 100x,直接比 COCO SOTA 不诚实。

- Flickr1K 只有 900 train,规模远小于 COCO/Flickr30k;LLaVA v1.5 官方 pretrain 是 558K pairs,我们训练量小约 100x。
- 20.59% 与经典 Show-Attend-Tell 在 Flickr30k 的 ~19-20% 属同一量级,足以证明**方法和 pipeline 正确**,但不宣称达到 SOTA。
- 诚信底线是贯穿项目的原则:5060 Ti 的数据绝不标成 Orin;能实测就标实测,不能就标预估。

---

## 组 D · 混合精度量化(JD 直接命中)

### D1. 你打算怎么量化?为什么量化对端侧 LLM 是刚需?

**结论:** 走 torchao(int8/int4 weight-only)+ GGUF Q4 两条 Orin 兼容路线;因为端侧单路推理是 memory-bandwidth-bound,压权重体积 ≈ 直接提速。**[计划,D3.2]**

- **原理:** batch=1 自回归时,每生成一个 token 都要把**整套权重**从显存过一遍,瓶颈是内存带宽而非算力。权重砍到 int4(体积 ~-75%),带宽压力近似减半,理论 tok/s 近似翻倍。
- **两条路线:**(1)`torchao int8_weight_only / int4_weight_only`,可移植;(2)`GGUF Q4_K_M` + llama.cpp,`.gguf` 硬件无关、Jetson 友好。
- **产出目标:** 一张「体积压缩比 × BLEU 掉多少 × 推理显存」对比表 + 一条 fp16→int8→int4 曲线,结论一句话(如「int4 体积 -75%,corpus BLEU 掉 X 点」)。**[计划]**

### D2.(JD 关键词)「混合精度量化」在你项目里具体指什么?

**结论:** 视觉塔 + projector 保 bf16,只把 LLM 主体压到 int4——按算子敏感度分配精度。

- LLM 是权重和计算主体,量化收益最大;projector 只有 3.94M,量化收益微乎其微,保 bf16 反而稳。
- 视觉塔一次性 encode,不在自回归热路径上,保 bf16 不影响吞吐但保精度。
- 这正对上「混合精度」字眼:讲得清**敏感度分析**(哪层能压、哪层不能压、为什么),而非无脑全量化。**[计划,进阶项]**

### D3. 量化后精度怎么评?怎么保证跨平台一致?

**结论:** 量化后用同一套 `evaluate.py` 重跑 corpus BLEU-4 定量、外加定性对比;GGUF 权重硬件无关,跨平台精度一致、只有速度差异。

- 定量:量化前后跑同一 100 张 val,比 corpus BLEU-4 掉多少点。
- 定性:挑 ~20 张图做量化前后 caption 对比。
- 跨平台一致性:同一份 `.gguf` 在 4060 / Orin / Xavier 都能加载,精度输出一致(仅极小 rounding 差异),速度差异符合带宽比例。差异不在权重本身,而在 llama.cpp 各后端实现。

### D4. 量化方案里踩过/避开了哪些坑?(Orin 红线)

**结论:** 严守 Orin NX(Ampere sm_87)兼容红线——禁 bitsandbytes / fp8 / fp4 / transformer_engine / Blackwell-only kernel。

- **bitsandbytes:** aarch64(Jetson)支持差,直接排除。
- **fp8/fp4:** 超出 Ampere 能力,不用。
- 量化只走 GGUF Q4(llama.cpp)或 torchao(可移植);任何新依赖先过一遍「Orin 兼容审计」再引入。
- torchao int4 的 tinygemm kernel 在 Ampere 上可用性需先确认。**[计划,排雷项]**

---

## 组 E · 算子性能调优 + 异构调度运行时(JD)

### E1.(JD 关键词)端侧推理的性能瓶颈在哪?怎么定位?

**结论:** 两段瓶颈——CLIP 一次性 encode(576 token,首 token 前的固定开销)+ LLM 自回归 decode(KV-cache / 带宽主导)。计划做 latency 分解定位。

- **prefill 侧:** CLIP encode 576 token 是首 token 延迟的一次性成本,与生成长度无关。
- **decode 侧:** 每 token 要过一遍权重 + 不断增长的 KV-cache,batch=1 时是内存带宽主导,不是算力主导。
- **调优计划:** 拆开 CLIP-encode vs LLM-decode 分别计时;对比 KV-cache on/off、SDPA vs eager attention 的差异,定位真实瓶颈。**[计划,A2.1]**

### E2. 为什么用内存带宽而不是 TOPS 来做迁移性能预估?

**结论:** 机载单路推理 batch=1 是 memory-bandwidth-bound,TOPS 只在 batch≥16 才主导。

- 每生成一个 token 都要把全部权重从显存读一遍;算力大量闲置,瓶颈在「权重过一遍要多久」。
- 因此按内存带宽比例能较好地估相对 tok/s:4060 是 272 GB/s,Orin NX 16G 是 102 GB/s(约 40%),Orin Nano 8G 68 GB/s,Xavier NX 51 GB/s。**[规格]**
- 据此的 tok/s:4060 实测 30-50 **[实测,约值]**;Orin NX 15-25、Orin Nano 8-15、Xavier 5-10 均为**[预估]**,需真机 rerun。

### E3.(JD 关键词)迁到 DJI 自研 NPU,异构调度这块你怎么想?

**结论:** 三级迁移路径,前两级技术全通用,第三级是「换编译工具链 + 换 runtime 调度」的目标 SDK hands-on。

- **4060 → Orin(Ampere):** 全 CUDA 生态,代码零改,速度按带宽比例下降。验证 pipeline 在端侧硬约束下稳定。
- **Orin → 自研 NPU:** 需换编译器(通常 MLIR-based,PyTorch → ONNX → 自研 IR)、量化走目标 SDK、算子级 fallback 策略(GELU / Attention 是否原生支持)、runtime 换 custom scheduler + DMA 管理。
- **我们架构的天然优势:** 冻结主干 + 3.94M adapter,主体权重可**离线一次性转换**,天然适配任意异构后端的「离线编译 + 上机调度」模型。**[叙事,基于同类 NPU 通用经验]**

---

## 组 F · 深度学习编译器 + 硬件迁移(JD)

### F1.(JD 关键词)「深度学习编译器」你能讲到什么程度?

**结论:** 定位为「讲得透 > 必须手做」,用 GGUF/llama.cpp 和 PyTorch→ONNX→TensorRT 两条链路作具象例子。

- GGUF/llama.cpp:讲清**图编译 + 算子融合 + 量化 kernel**是怎么把一张计算图落到具体后端的。
- PyTorch → ONNX → TensorRT/自研 NPU:讲清导出、图优化、engine 构建这条编译-部署链路。**ONNX 导出本身可移植,engine 构建是目标平台特定的**(诚实说明)。
- 不硬做 torch.compile:它在 Jetson 兼容性存疑,踩 Orin 红线,收益低。**[定位:talk-shop,不强求代码产物]**

### F2. 从 4060 到 Orin 的兼容性,你审计过哪些点?

**结论:** Orin NX(Ampere sm_87)与开发平台架构同系,bf16 权重零转换、代码 100% 兼容,审计已通过。

- bf16 / FlashAttention v2 / 标准算子(SDPA、LayerNorm、softmax、GELU、GEMM)Orin 全支持,无超 Ampere 特性。
- 同一份代码已在 4060(Windows)、5060 Ti(Ubuntu)两平台零改动跑通,是「可迁移」的实锤。**[实测]**
- Orin NX 16G 是最优目标:16G LPDDR5、102 GB/s 带宽、100 TOPS INT8,对齐大疆 Matrice 4 系列主控算力段,DevKit ~¥4300。

### F3. 8GB 部署上限是怎么守的?为什么这是核心指标?

**结论:** 推理显存锁定 1.6GB、稳进 8GB,是「Jetson-class 可部署」最诚实、最可迁移的证据。

- 推理峰值 **1.6GB [实测]**,连 Orin Nano 8G 都容得下,留足系统 + 其他任务余量。
- 16G 的 5060 Ti 只用于**提升训练迭代效率**,绝不作为交付叙事卖点——训练要 9.9GB,但交付的是推理可部署性。
- 推理显存 = 端侧可部署性的关键指标,比训练指标更贴近 JD 的真实场景。

---

## 附:诚信标注速查

- **[实测]:** corpus/sentence BLEU、baseline 对比、训练/推理显存、参数量、loss 曲线、跨平台零改动跑通、base→Instruct 的 BLEU 变化。
- **[预估]:** 所有非 4060/5060Ti 的 Jetson tok/s、Xavier fp16 精度掉幅。
- **[计划]:** 量化全套(D3.2)、混合精度敏感度分析、latency 分解、ONNX 导出、自研 NPU 迁移。
</content>
</invoke>
