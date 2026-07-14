# scratch-vlm 规划 · ROADMAP

**Date:** 2026-07-13
**结构:** 总 → 分。先定总目标与贯穿约束,再逐层拆到可执行任务。
**调整原则:** 每个下级任务都服务于某个上级目标。若某个细节任务证明"不成"(太难/收益低/踩 Orin 红线),**不硬做**——退回其上级目标,用替代手段或"talk-shop 讲清楚"来满足它。

---

## 0 · 总目标

**拿下 DJI 端侧 AI 系统工程师岗(2027 提前批)。**

本项目是服务于这一目标的作品级 demo。成功判据:**面试中能用硬数据 + 可信叙事,覆盖 JD 的每一条关键词**,且经得起追问。

JD 关键词(所有技术选择的锚):深度学习编译器 · 混合精度量化 · 异构调度运行时 · 算子性能调优 · 端侧大模型。加分:机载机器人 AI 部署 · RoboMaster 经验。

---

## 1 · 贯穿约束(所有工作先过这几条闸)

1. **8GB 部署上限**:交付/声称的模型推理必须 ≤ 8GB(Jetson-class 可部署的可信证据)。16G 仅用于提升训练迭代效率,**绝不作叙事卖点**。
2. **Orin NX 兼容红线**:目标部署平台 = Jetson Orin NX 16G(Ampere sm_87)。禁止引入超 Ampere 特性:❌ bitsandbytes(aarch64 支持差)、❌ fp8/fp4、❌ transformer_engine、❌ Blackwell-only kernel。量化只走 **GGUF Q4(llama.cpp,Jetson 友好)或 torchao(可移植)**。任何新工作先过兼容审计。
3. **诚信底线**:不虚标硬件数据(5060 Ti 数据不标 Orin);指标标清 corpus vs sentence BLEU;非实测一律标"预估";声称的模型必须本地真有、真训过。
4. **深度 > 广度**:每个支柱选 1-2 个方向做透,不铺开。

---

## 2 · 三大支柱(支撑总目标)

```
总目标: 拿下 DJI 端侧 AI 系统岗
├── 支柱 A · 技术深度   (硬覆盖 JD 关键词)      ← 项目的"肉"
├── 支柱 B · 叙事与诚信 (可信的端侧迁移故事)    ← 项目的"骨"
└── 支柱 C · 面试交付物 (demo / Q&A / pitch)    ← 项目的"脸"
```

---

## 3 · 支柱 A · 技术深度(JD 硬覆盖)

> 优先级:A1 量化(JD 最硬命中)> A2 推理/算子优化 > A3 编译器视角(最难,偏 talk-shop)。

### A1 · 混合精度量化 【最高优先 · JD 直接命中】
- **目的**:命中"混合精度量化 + 端侧大模型";产出硬件无关的"体积 × 精度"证据(这是端侧最诚实、最可迁移的卖点)。
- **动作**:
  - A1.1 权重量化 LLM:torchao `int8_weight_only` / `int4_weight_only`(Qwen 是权重/计算主体)。守 Orin 兼容。
  - A1.2 量化后用 `evaluate.py` 重跑 **corpus BLEU-4**,画曲线:fp16 → int8 → int4 的**体积压缩比 vs BLEU 掉多少 vs 推理显存**。
  - A1.3 (进阶)**混合精度**:视觉塔/projector 保 bf16,LLM 压 int4 —— 直接对上"混合精度"字眼,讲敏感度分析。
  - A1.4 (进阶,edge 具象)GGUF Q4 + llama.cpp:LLM 导出 Q4_K_M,量 tok/s;注明全 pipeline 需 mmproj(CLIP+projector)另接,作为"部署路径"说明。
- **验收**:一张量化对比表(精度/体积/显存/速度)+ 一条曲线图;结论一句话("int4 体积 -75%,corpus BLEU 掉 X 点")。
- **成本**:torchao 路径 ~0.5-1 天;GGUF ~1-2 天。
- **风险/红线**:严禁 bitsandbytes;torchao int4 tinygemm 在 Ampere 可用需确认。

### A2 · 推理 / 算子性能调优 【次优先 · 命中"算子性能调优"】
- **目的**:命中"算子性能调优";量化端侧推理瓶颈,支撑 migration_analysis 的 roofline 论述从"预估"走向"实测方法学"。
- **动作**:
  - A2.1 **latency 分解**:拆 CLIP-encode(576 tokens,首 token 前的一次性开销)vs LLM 自回归 decode(KV-cache 主导),分别计时,定位瓶颈。
  - A2.2 (可选)KV-cache on/off、SDPA vs eager 的量化对比。
  - A2.3 (进阶,Orin 友好)CLIP vision tower 导出 **ONNX**(→ Jetson 上转 TensorRT 的第一步)。engine 构建是目标平台特定的(诚实说明),但 ONNX 导出本身可移植,展示部署路径。
- **验收**:分阶段 latency 表 + 瓶颈定位结论。
- **成本**:A2.1 ~0.5 天;A2.3 ~1 天(有踩坑风险)。

### A3 · 编译器视角 【最低优先 · 命中"深度学习编译器" · 偏 talk-shop】
- **目的**:JD 有"深度学习编译器",但这块手做成本极高、且易踩 Orin 红线(torch.compile 在 Jetson 兼容性存疑)。定位为**能讲清楚 > 必须手做**。
- **动作**:把 GGUF/llama.cpp 的**图编译 + 算子融合 + 量化 kernel**作为具象例子讲透;把 PyTorch→ONNX→TensorRT/自研 NPU 的编译-部署链路讲清楚(migration_analysis 场景 C 已有骨架)。
- **验收**:talk-shop Q&A 里能覆盖;不强求代码产物。

---

## 4 · 支柱 B · 叙事与诚信

### B1 · 4060 vs 5060 Ti 迁移对比表 【收尾 · ~10 分钟】
- 把旧配置(CLIP-B/32 + SmolLM2, 300 步 batch=8)在 5060 Ti 上 apples-to-apples 跑一次,填 VRAM / wall-time / BLEU 三行 → "同一份代码零改动跨两平台"的实锤。写入 README。

### B2 · 迁移分析文档更新
- `benchmark/migration_analysis.md` 补:5060 Ti(Blackwell)实测项 + 新架构(CLIP-L/Qwen)的显存/latency;把能实测的"预估"字段升级为"实测"。

### B3 · 诚信审计维持
- 任何新工作(尤其量化)先过 Orin 兼容审计;指标继续 corpus/sentence 双报。

### B4 · 仓库清理 【需用户点头】
- 删除过时中间件:`projector_L14_qwen*.pt`(base 版,已被 Instruct 微调版取代;但 base_best 是微调的初始化来源,建议保留可复现);清理空壳模型目录 `AI-ModelScope--clip-vit-base-patch32`。**动 checkpoint 前确认。**

---

## 5 · 支柱 C · 面试交付物(临面试再密集做,避免返工)

### C1 · talk-shop Q&A(~20 问)
- 覆盖 JD 每条 + 本项目真实踩过的坑(极佳的"深度"素材):
  - base vs Instruct(冻结 LM 不会吐 `<|im_end|>` → 生成不终止 → BLEU 从 22% 掉到 12%)
  - bf16 vs fp16(数值稳定 + Ampere 原生 + Xavier 需转 fp16)
  - CLIP-L@336 的 576 tokens 为何让训练显存暴涨、为何推理仍只 1.6GB
  - 量化权衡(体积 vs 精度曲线)
  - corpus vs sentence BLEU、为何不与 COCO SOTA 直接比
  - 冻结主干 + 3.94M adapter(0.49%)的经济学

### C2 · Gradio demo
- 现场演示:上传图 → 生成 caption,展示实际推理 + 显存占用。

### C3 · 三档 pitch 稿
- 15 分钟(完整)/ 2 分钟(电梯)/ 30 秒(一句话)。

### C4 · CV 条目更新
- 更新简历项目条目为真实数字(corpus BLEU 20.59% / 3.94M 可训 / 端侧 1.6GB)。**注:主简历 `currbg.tex` 在另一仓库、未迁过来,改它需明确许可。**

---

## 6 · 可选支线清单(optional side-branches)

这些都**不在关键路径上**,是"锦上添花/按需触发"的独立支线。按叙事价值与成本排列。

| 编号 | 支线 | 目的 / 命中点 | GPU 需求 | 成本 | 叙事价值 |
|------|------|--------------|---------|------|---------|
| **S1** | 大数据集硬化 | 标准 split(Flickr30k/COCO)上可对标硬 BLEU,堵"数据集太小" | 重(下载+训练) | ~1-2 天 | 中 |
| **S2** ⭐ | **航拍/DJI 领域适配**(已确认要做,**排最后**) | VLM 在无人机航拍图上 caption/微调,直连 DJI 主业,"通用 demo"→"懂你们场景" | 重 | ~1-2 天 | **最高** |
| **S3** | VQA 扩展 | 从 captioning 扩到视觉问答,证明 pipeline 泛化性 | 中 | ~1 天 | 中 |
| **S4** | 设计消融实验 | CLIP-B/32 vs L/14、SmolLM2 vs Qwen、projector 层数;"论证过每个选择"的深度素材 | 重 | ~1-2 天 | 中高 |
| **S5** | 竞品 / benchmark 调研 | 同类 tiny/edge VLM 怎么定位、标准榜怎么报;喂叙事 | 无 | ~0.5 天 | 中 |
| **S6** | 量化方案技术调研 | torchao vs GGUF vs GPTQ/AWQ × Orin 兼容 → 决策备忘;**给主线 A1 排雷** | 无 | ~0.5 天 | (服务主线) |

**触发原则**:
- **S2(航拍)已确认要做,但优先级排最后**——放在所有主线(P1-P4)之后,作为"领域化收尾/capstone"。它叙事价值最高,但依赖前面的技术+系统+交付都已扎实;��且 GPU 重、需备航拍数据,所以压到最后、时间允许就做。
- S1/S4 是"模型硬化"投资,按需触发(S1 见"堵数据集太小",S4 见"深度素材");同样靠后。
- S5/S6 是纯调研、不吃 GPU,可在主线跑 GPU 时**提前并行**(见 §7bis)。

## 7bis · 并行执行模型(subagent 使用约束)

**现实约束:本机只有一张 GPU。** subagent 共享同一张卡 → **训练/评测/量化跑等 GPU 活无法真并行**(抢显存)。因此:

- ✅ **subagent 适合并行的(非 GPU)**:调研(S5/S6、DJI 岗)、文档起草(talk-shop Q&A、pitch)、代码脚手架(量化脚本、gradio app、ONNX 导出脚本——先写好,GPU 活稍后串行跑)。
- ❌ **必须主线串行(GPU)**:训练、评测、量化实测、latency profiling。

**最优打法**:subagent 并行铺"调研 + 起草 + 写脚本",主线串行占用 GPU 跑训练/评测。**所有 subagent 由用户明确 "动手" 后再启动。**

---

## 7 · 执行顺序(分阶段)

| Phase | 内容 | 估时 | 说明 |
|-------|------|------|------|
| **P1 · 收尾** | B1 迁移对比表 + B2 文档更新 + B4 清理 | ~0.5 天 | 彻底关闭迁移线 |
| **P2 · 技术核心** | A1 量化全套 + A2.1 latency 分解 | ~1-2 天 | JD 最硬命中,项目的"肉" |
| **P3 · 进阶(可选)** | A1.3 混合精度 / A2.3 ONNX / A3 编译器视角 | ~1-2 天 | 按余力和面试时间挑 |
| **P4 · 交付包装** | C1 Q&A + C2 demo + C3 pitch + C4 CV | ~1 天 | 临面试密集做 |
| **P5 · 领域化收尾** | **S2 航拍/DJI 领域适配**(capstone) | ~1-2 天 | 已确认要做,**排最后**;时间允许就做,叙事价值最高 |
| (按需支线) | S1 大数据集 / S4 消融 | ~1-2 天 | 仅在需要标准对标 / 深度素材时触发 |
| (可并行调研) | S5 竞品调研 / S6 量化方案调研 | ~0.5 天 | 不吃 GPU,主线跑 GPU 时并行 |

---

## 8 · 附:已完成(截至 2026-07-13)

**前期(架构升级)**
- ✅ 迁移 Ubuntu/5060 Ti + 3 项一致性验证
- ✅ 架构升级 CLIP-L/14@336 + Qwen2.5-0.5B-Instruct,重训 → **corpus BLEU-4 20.59%**(旧 SOTA ~17%),推理 1.6GB
- ✅ Orin NX 兼容性审计通过(sdpa / bf16 / 标准算子,无超 Ampere 特性)
- ✅ 跨平台 ckpt 路径 fallback + ChatML eos 修复;train.py 加 grad-accum / init-projector 微调
- ✅ evaluate.py corpus + sentence 双 BLEU;README + setup_env 全面对齐(声称=现实)

**P1 收尾(✅ 完成)**
- ✅ B1 迁移对比表:5060 Ti 训练 28.2s(vs 4060 5.8min)、VRAM 2.63GB(一致)、4060 ckpt 跨平台复评 16.98%(≈原 17.15%)→ 写入 README
- ✅ B2 migration_analysis.md 更新:5060 Ti 实测 + 新架构显存 + latency 分解

**P2 技术核心(✅ 完成)**
- ✅ A1 量化:torchao int8(near-lossless, -14% 显存)/ int4(-26% 显存, -4.6 BLEU);天然混合精度;Orin 红线实战(mslk→tile_packed_to_4d)→ `docs/quantization_plan.md` + README
- ✅ A2.1 latency 分解:视觉编码 15.3ms(一次性)+ prefill 14.1ms + decode 7.29ms/tok(137 tok/s)→ `benchmark/profile_latency.py`

**P3 进阶(✅ 完成)**
- ✅ A1.3 混合精度:weight-only 只压 LLM、CLIP/projector 守 bf16,天然达成
- ✅ A2.3 ONNX:视觉前端(CLIP-L@336+projector)导出 ONNX + onnxruntime 验证一致 → `benchmark/export_onnx.py`
- ✅ A3 编译器视角:并入 talk-shop Q&A(F 组)

**P4 交付包装(✅ 完成)**
- ✅ C1 talk-shop Q&A(21 问)→ `docs/talkshop_qa.md`
- ✅ C2 Gradio demo(图→caption + 显存/latency + 量化开关)→ `app.py`
- ✅ C3 三档 pitch → `docs/pitch.md`
- ✅ C4 CV 条目草稿(中英)→ `docs/cv_entry.md`(合入 currbg.tex 待用户许可)
- ✅ 调研支线:S5 竞品定位 → `docs/competitor_benchmark.md`;S6 量化方案 → `docs/quantization_plan.md`

**S1 数据集硬化(✅ 完成 · 2026-07-14, 换源)**
- 完整 Flickr30k(nlphuji)图像 zip 走 xethub CDN 被限流下不动 → **换 `jxie/flickr8k`(parquet,普通 CDN)**。
- 从零训练于 Flickr8k train(6k 图)、标准 1k test(5 参考)评测 → **corpus BLEU-4 31.76%**,明显超 Show-Attend-Tell Flickr8k ~19.5-21.3。**修掉了原 Flickr30k-test 上"训练用到 test"的方法学问题**,是最干净可对标的数字。脚本 `data/prepare_flickr8k.py`,ckpt `projector_flickr8k_best.pt`。

**P5 领域化收尾(⏸ 用户暂缓 · 2026-07-14)**
- ⏸ S2 航拍/DJI 领域适配:执行 spec 已就绪(`docs/s2_aerial_plan.md`,一条命令可执行)。用户决定暂缓,等有真实航拍数据或面试临近再启动;届时只差数据集 green-light。
