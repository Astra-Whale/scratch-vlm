# AGENT_CROSSCHECK · 项目状态与交叉检查手册

**Date:** 2026-07-14
**用途:** 供其他 agent **独立交叉检查**本项目的现状与每一条硬数据。本文件强调:(a) 每个数字**从哪来、怎么复核**;(b) 易混淆点;(c) 已知坑。规划见 [`process/ROADMAP.md`](process/ROADMAP.md),历史迁移见 [`process/AGENT_HANDOFF.md`](process/AGENT_HANDOFF.md)。

---

## 0 · 交叉检查者请先读这条

**最省事的验证路径(无需 GPU / 模型 / 重新下载数据):**
仓库已提交 `logs/eval_flickr8k_qwen3_test_1000.json`(Qwen3 旗舰,1000 条 `{image, gen, refs}`)。直接:
```bash
conda activate dl
python benchmark/eval_coco_metrics.py --json logs/eval_flickr8k_qwen3_test_1000.json --no-spice
```
应复现:**CIDEr 0.9403 · BLEU-4 32.91 · BLEU-1 75.77 · METEOR 0.2763 · ROUGE-L 0.5727**。
这直接验证了旗舰指标,不依赖 checkpoint 或 flickr8k 图像(两者都 gitignored)。
> 旧 Qwen2.5 轨的 eval JSON(CIDEr 0.9268 / BLEU-4 31.73)已随试错日志归档到本地 `logs/_archive/eval_flickr8k_test_1000.json`(移出 git 跟踪);**旗舰口径已迁 Qwen3**;Qwen2.5/SmolLM2 ckpt 已归档 `checkpoints/_archive_non_spec/`。

---

## 1 · 项目一句话

面向 **DJI 端侧 AI 系统工程师岗** 的作品级 demo。LLaVA v1.5 风格:**冻结** CLIP-ViT-L/14@336 + **冻结** Qwen3-0.6B + **可训** 2 层 MLP projector(1024→2048→1024,**4.20M,占总 ~904M 的 0.46%**)。卖点是端侧全流程(训练→推理→量化→迁移)+ 参数效率,不是刷 captioning 榜。(Qwen2.5/SmolLM2 早期轨已归档 `checkpoints/_archive_non_spec/`。)

## 2 · 环境

- conda env **`dl`**:Python 3.11,**torch 2.11+cu128**,transformers 5.13,torchao 0.17,pycocoevalcap,pyarrow。
- 硬件:RTX 5060 Ti 16G(Blackwell **sm_120**)。Blackwell 需 torch ≥ 2.7 / CUDA 12.6+。
- 离线跑加 `HF_HUB_OFFLINE=1`。模型:CLIP-L/14@336(HF cache)、**Qwen3-0.6B 本地 `weights/Qwen3-0.6B/`**;(归档轨)Qwen2.5-0.5B-Instruct、CLIP-B/32、SmolLM2-360M。
- `conda run -n dl` 会缓冲 stdout;要实时日志加 `PYTHONUNBUFFERED=1`。

## 3 · 权威硬数据(⚠️ 注意每行来自哪个数据集/ckpt)

### 3a · 旗舰:干净可对标(Flickr8k proper split)· **Qwen3-0.6B**
- **ckpt**:`checkpoints/projector_stage1_qwen3_best.pt`(**从零训练**于 Flickr8k train 5999 图,非 init;LLM=Qwen3-0.6B)
- **eval**:标准 Flickr8k **test 1000 图 × 5 参考**(`logs/eval_flickr8k_qwen3_test_1000.json`)
- **官方 pycocoevalcap**:**CIDEr 0.940(×100=94.0) · BLEU-4 32.91 · BLEU-1 75.77 · METEOR 0.276 · ROUGE-L 0.573**
- 手写 corpus BLEU-4 **32.93%**(与官方 32.91 差 0.02 → 手写口径已被官方复核)
- 对标:Show-Attend-Tell Flickr8k BLEU-4 ~19.5/21.3 → **超出约 11-13 点**(同数据集/指标/参考数)
- 归档对照:旧 Qwen2.5 轨(`_archive_non_spec/projector_flickr8k_best.pt`)同口径 CIDEr 0.927/BLEU-4 31.73;Qwen3 hidden 1024>896 略增容量后小幅提升。

### 3b · 架构开发期(⚠️ Flickr30k test-1k 子集,train-on-test,已被 3a 取代作对标;Qwen2.5,**已归档**)
> §3b/3c/3d 均为 **Qwen2.5-0.5B-Instruct arch-dev 轨**,ckpt 已移入 `checkpoints/_archive_non_spec/`。量化的 spec 对齐交付以 §10 的 **GGUF Q4_K_M(Qwen3)** 为准;此处 torchao int8/int4 数字仅作早期探索记录。
- **ckpt**:`_archive_non_spec/projector_L14_qwenInstruct_ft_best.pt`
- **数据**:`data/flickr_1k/` 实为 **Flickr30k Karpathy test 的 1000 图**,900/100 内部切分(**训练用到了 test**)
- corpus BLEU-4 **20.59%** / sentence 22.18%(100 val)。**仅作架构演进记录,勿用于论文对标。**

### 3c · 量化(⚠️ 在 3b 的 arch-dev ckpt + Flickr30k-test-100 上测,非 Flickr8k)
| quant | corpus BLEU-4 | 权重常驻显存 |
|---|---|---|
| bf16 | 20.59% | 1536 MB |
| int8(排除 lm_head) | 20.04% | 1191 MB (-22%) |
| int4(排除 lm_head,tinygemm) | 18.33% | 1057 MB (-31%) |
- 混合精度:只压 Qwen transformer Linear,CLIP/projector/lm_head 守 bf16。详见 [`quantization_plan.md`](quantization_plan.md)。

### 3d · latency / 迁移(arch-dev ckpt)
- 推理 batch=1:**1.6 GB**;视觉编码 15.3ms(一次性)+ prefill 14.1ms + decode 7.29ms/tok = **137 tok/s**(`logs/latency_profile.json`)。
- 迁移:5060 Ti 训练 300 步 **28.2s / VRAM 2.63GB** vs 4060 5.8min/2.67GB;4060 ckpt 跨平台复评 sentence **16.98%**(≈原 17.15%)。

## 4 · 如何复核每个数字

| 声明 | 复核命令 | 需要 |
|---|---|---|
| 3a 官方指标 | `python benchmark/eval_coco_metrics.py --json logs/eval_flickr8k_qwen3_test_1000.json --no-spice` | 仅提交的 JSON(**无需模型/数据**) |
| 3a 手写 BLEU | `python evaluate.py --ckpt checkpoints/projector_stage1_qwen3_best.pt --data data/flickr8k/test.jsonl --image-root data/flickr8k/images --max-samples 1000` | ckpt + flickr8k 数据(均 gitignored,需重建) |
| 3c 量化(归档轨) | `python evaluate.py --ckpt checkpoints/_archive_non_spec/projector_L14_qwenInstruct_ft_best.pt --quant int8 --max-samples 100` | 归档 ckpt + flickr_1k 数据 |
| 3d latency | `python benchmark/profile_latency.py` | ckpt |
| 数据重建(flickr8k) | 下 `jxie/flickr8k` 4 个 parquet 到 `data/flickr8k/` → `python data/prepare_flickr8k.py` | ~1.1GB 下载 |

> **注**:`checkpoints/*.pt` 与 `data/` blob 均 **gitignored**(未进仓库)。要跑需模型的复核,须先重建数据(§4 末行)+ 重训,或索取 ckpt。**无模型的复核走 §0。**

## 5 · 文件地图

**代码**:`model/`(vlm/vision_encoder/projector)· `train.py`(--grad-accum / --init-projector)· `evaluate.py`(corpus+sentence BLEU / --quant int8|int4 / 路径 fallback)· `inference.py` · `app.py`(Gradio)· `data/*.py`(prepare_flickr8k 等)· `benchmark/`(profile_latency / export_onnx / eval_coco_metrics)
**文档**:`README.md`(根)· `docs/`(CROSSCHECK / ALIGN_SELFSPEC / quantization_plan / benchmark_landscape / competitor_benchmark / talkshop_qa / pitch / cv_entry / data_sourcing / llamacpp_pipeline)· `docs/process/`(ROADMAP / AGENT_HANDOFF / MIGRATE_TO_UBUNTU / setup_env / s2_aerial_plan,试错/迁移/规划归档)· `benchmark/migration_analysis.md`
**证据**:`logs/`(训练日志 + 各 eval JSON,已提交)
**gitignored**:`checkpoints/`(权重)· `data/` blob · `weights/`(缓存)· `onnx/`

## 6 · 易混淆点(交叉检查重点核对)

1. **两套数据别混**:Flickr8k(3a,干净,旗舰)vs Flickr30k-test-1k(3b,train-on-test,旧)。`data/flickr_1k/` 名字叫 1k 但**是 Flickr30k 的 test split**。
2. **量化/latency 数字在 arch-dev ckpt(3b/3c/3d)上测,不是 Flickr8k**。若要 Flickr8k 上的量化数需另跑。
3. **CIDEr 跨数据集不可比**:Flickr8k 的 94.0 ≠ COCO SOTA 130-155(不同语料 TF-IDF)。
4. **BLEU 两种口径**:corpus(标准、可对标)vs sentence-level+平滑(内部追踪,偏高)。
5. **手写 BLEU 已被官方 pycocoevalcap 复核**(Qwen3: 32.93 vs 32.91;归档 Qwen2.5: 31.76 vs 31.73)。

## 7 · 已知坑(踩过并修复,供核对代码)

1. **Qwen base ≠ Instruct**:base 冻结 LM 不吐 ChatML `<|im_end|>` → 生成越过 caption 吐垃圾 → BLEU 22%→12%。必须用 **Instruct**;`model/vlm.py::generate` 已把 `<|im_end|>` 加进 eos。
2. **torchao int4 默认 packing 需 `mslk`(非便携,撞 Orin 红线)** → 用 `int4_packing_format="tile_packed_to_4d"`(tinygemm,Ampere 原生)。见 `evaluate.py`。
3. **量化 tied lm_head 反效果**:Qwen `tie_word_embeddings=True`,量化 lm_head 打破共享(+136MB)且伤精度 → `filter_fn` 排除 lm_head。
4. **完整 Flickr30k(nlphuji)zip 走 xethub CDN 本环境限流下不动** → 换 `jxie/flickr8k`(parquet 普通 CDN)。
5. **fp16 + 随机 projector 在 test_forward 偶发 loss=NaN**(cosmetic);bf16 稳定,项目全程 bf16。
6. **跨平台 ckpt 路径**:旧 ckpt 存 Windows 绝对路径;`evaluate.py`/`inference.py` 仅对**绝对/盘符路径**做本地 fallback(HF repo_id 不碰)。

## 8 · 未完成 / 暂缓 / 待人工

- **S2 航拍领域适配**:**已撤销**(2026-07-14 用户决定不做);历史 spec 仍留 `docs/process/s2_aerial_plan.md` 备查,不再作为待办。
- **git 身份是占位** `徐悦 <xuyue@localhost>`(本地配置),需改真实邮箱。
- **SOTA ckpt gitignored**(`*.pt`);若要开箱可跑可 `git add -f checkpoints/projector_stage1_qwen3_best.pt`(8.4MB)。
- **无 git remote**;未推送。
- **SPICE 未跑**(需大内存 Java、1000 图易卡);CIDEr 已是主指标。
- **COCO 同轴对标未做**(需 COCO 训练,大成本且注定不 competes SOTA;结论:不比数字,比范式+效率,见 `docs/benchmark_landscape.md`)。

## 9 · 记忆锚点

Agent 记忆库(`.claude/.../memory/`)有 `vlm-strategy-constraints`(8GB 上限 / Orin 红线 / SOTA / 量化要点)与 `vlm-migration-findings`。与本文一致;若冲突以本文 + 仓库实测为准。

---

## 10 · selfspec 对齐执行状态(2026-07-14 · 压缩前快照)

**已完成**:#4 换 Qwen3-0.6B(本地 `weights/Qwen3-0.6B/`,curl 续传绕过 hf-xet 挂死);stage-1 projector 对齐(`checkpoints/projector_stage1_qwen3_best.pt`,Flickr8k,eval caption 干净、`<think>` 已在 vlm.py 剥离);#26 窄核心(llama.cpp Qwen3 GGUF **Q4_K_M 0.48GB/3.12×**,PPL f16 19.63→Q4 21.35,llama-server SSE 流式跑通,见 `docs/llamacpp_pipeline.md`)。

**进行中**:stage-2 LoRA SFT(`train_sft.py`,batch=2/accum=8/steps=300,输出 `checkpoints/vlm_stage2_lora/{projector.pt, lora_adapter/}`)。注:batch=4 会 OOM,必须 batch≤2。

**下一步(stage-2 完成后立即)· POPE 幻觉对比 #25**:
```
# stage-1 (无 LoRA)
python benchmark/evaluate_pope.py --projector-ckpt checkpoints/projector_stage1_qwen3_best.pt \
    --pope-dir data/pope --image-root data/coco/val2014 --out logs/pope_stage1.json
# stage-2 (加 LoRA)
python benchmark/evaluate_pope.py --projector-ckpt checkpoints/vlm_stage2_lora/projector.pt \
    --lora-adapter checkpoints/vlm_stage2_lora/lora_adapter --pope-dir data/pope --image-root data/coco/val2014 --out logs/pope_stage2.json
```
对比两者 F1 → 简历 #17"两阶段对幻觉抑制的边际收益"。

**之后待办**:多模态 mmproj 集成(CLIP+projector→llama.cpp,我自己弄,S6 已标 fiddly);更新 README/ALIGN_SELFSPEC 反映对齐结果;`git add -A && commit`(新增 train_sft.py/sft_dataset.py/evaluate_pope.py/fetch_coco_images.py/docs 等尚未提交)。数据/ckpt 均 gitignored。

### §10 更新(POPE 结果 + 待 fix)· 2026-07-14
- **两阶段训练完成**:stage-2 SFT 300步/loss 3.26→1.76,产物 `checkpoints/vlm_stage2_lora/{projector.pt, lora_adapter/}`(adapter 602M 因含 resize 的 embedding)。
- **POPE 跑通但结果无效**:stage-1 & stage-2 均 acc=50/f1=0/yes%=0(`logs/pope_stage{1,2}.json`)。**模型从不答 yes**。
- **根因(待验证)**:SFT 只用了 `detail_23k`(纯描述),没教 yes/no QA → 模型被问"有没有X"时描述图像而非答 yes/no,`evaluate_pope.py::parse_yesno` 默认判 no → 全 no。
- **FIX(压缩后第一步)**:
  1. 先验证:加载 stage-2 对 2-3 条 POPE 问题 `generate`,打印 raw 输出,确认是"描述/乱答"而非 parser bug。
  2. 下 `conversation_58k.json`(LLaVA-Instruct 多轮 QA,含 yes/no)+ 补对应 COCO 图 → 重跑 stage-2 SFT(混 detail+conversation),使模型学会短答 yes/no。
  3. 重跑 POPE → 得有效 stage-1 vs stage-2 F1 对比(#17)。
- POPE 之外的对齐项(#4/#6-10/#26 窄核心)均已成;剩 mmproj + 文档 + git。

### §10 更新 2(根因已验证 + 混合数据修复)· 2026-07-14
- **根因证实(非假设)**:加载 stage-2 对 POPE 前 5 题 `generate` 打印 raw:模型对 "Is there a skis in the image?"(GT=yes)输出 `'The image features a person skiing down a snowy slope...'`。→ **模型看对了(确实识别出滑雪),但输出的是描述而非 yes/no 短答**;`parse_yesno` 保守判 no。**确认是指令跟随/输出格式 gap,非视觉能力、非 parser bug。**
- **修复(方法对齐 LLaVA v1.5)**:下 `conversation_58k.json`(126MB,LLaVA-Instruct 通用视觉 QA;curl 走 hf-mirror,`hf_hub_download` head 调用失败故绕过)。与本地 3000 张 train2014 图交集 = **2002 条**(零额外下载,两个 split 同抽自 COCO train2014 池)。合并 detail(3000)+conversation(2002)=**5002 条** → `data/llava_instruct/sft_mix.json`。从 **stage-1 projector 重新初始化**重跑 stage-2(不接续无效的纯描述 stage-2),输出 `checkpoints/vlm_stage2_mix/`。
- **训练配置**:batch=1/accum=16(有效 16)/steps=500/lr 2e-4/lora r=16。**为何 batch=1**:conversation 多轮样本长(截到 max_length=1024),加上 CLIP-L 的 **576 视觉 token**,单样本有效序列 ~1600 token → batch=2 峰值 15.3GB 濒临 OOM;batch=1 峰值 ~14.9GB 且被 max_length 硬顶(高水位不再涨),可靠跑完。
- **⚠️ 诚信红线(交叉检查者重点核对)**:**绝不合成 POPE 同款"图里有没有 {物体}?→yes/no"探针来训练**。那等于在测试集分布上训练,POPE F1 会变成"背没背下格式"而非幻觉度量 → 简历 #16/#17 声称即造假。POPE 必须保持**零样本留出**;yes/no 能力只能来自**通用** QA 指令微调(conversation_58k),这正是 LLaVA v1.5 的做法(用通用 VQA/对话训练,零样本上 POPE)。合成通用视觉指令数据本身不违规(LLaVA-Instruct 本就是 GPT-4 合成),违规的是合成**与测试同格式**的探针。
- **下一步**:训练完 → 重跑 POPE(stage-1 vs vlm_stage2_mix)→ 有效 F1 对比。

### §10 更新 3(POPE 达成 + 数据消融链)· 2026-07-14
**最终有效结果 —— `checkpoints/vlm_stage2_mix2/`(detail 3000 + conversation 2002 + 平衡VQA 6020 = 11022 条,700步,peak 7.2GB)**:

| split | acc | precision | recall | f1 | yes% | n |
|---|---|---|---|---|---|---|
| random | 82.9 | 81.14 | 85.73 | 83.37 | 52.83 | 3000 |
| popular | 76.1 | 71.88 | 85.73 | 78.20 | 59.63 | 3000 |
| adversarial | 70.2 | 65.41 | 85.73 | 74.21 | 65.53 | 3000 |
| **avg F1** | | | | **78.59** | | |

`logs/pope_stage2_mix2.json`。**random>popular>adversarial 单调退化 = POPE 正确行为**;yes% 从 100→53(random)偏置消除。

**数据消融链(#17 真正卖点,honest)**:

| SFT 数据 | ckpt | POPE 行为 | acc | avg F1 |
|---|---|---|---|---|
| 仅 detail | `vlm_stage2_lora`(旧) | 全 no(不会QA) | 50 | 0 |
| +conversation | `vlm_stage2_mix` | 全 yes(训练数据93%yes) | 50 | 66.67(虚高) |
| +平衡VQA | `vlm_stage2_mix2` | 均衡 | 70–83 | **78.59** |

**结论**:模型幻觉/作答行为随训练数据分布走;平衡 yes/no 是抑制过度肯定的关键杠杆。
**诚信守住**:POPE 零样本(训练 train2014 / POPE val2014,图无重叠);**从未**训练 POPE 同款物体存在探针。VQAv2 平衡 yes/no 是 LLaVA v1.5 正宗配方。
**stage-1 对照**:projector-only 无指令跟随能力 → all-no/f1=0(`logs/pope_stage1.json`),印证两阶段设计中"对齐必要但不充分,指令微调才产生能力"。

### §10 更新 4(mmproj 多模态集成端到端跑通)· 2026-07-15
- **#11 CLIP+projector 打包 + 端侧多模态推理达成**:自训视觉栈打包成 llama.cpp mmproj GGUF,与 LoRA 合并后 Qwen3 GGUF 组合,端到端图文推理跑通。
- **产物**:`weights/gguf/qwen3-stage2-merged-q4_k_m.gguf`(372MiB,LoRA 已合并)+ `weights/gguf/mmproj-model-f16.gguf`(590MB,CLIP-L f16 + projector);工具 `tools_merge_lora.py`。
- **方法**:LoRA merge_and_unload → GGUF → Q4_K_M;CLIP+projector 走 legacy `convert_image_encoder_to_gguf.py`(projector `linear1→mm.0`/`linear2→mm.2`,`--projector-type mlp`)。脚本默认丢 CLIP 末层到 `blk.22`,与 `VisionEncoder(select_layer=-2)` **特征层严格一致**。
- **验证**(`llama-mtmd-cli`,纯 CPU,`--temp 0`,火车图 COCO_val2014_000000001171):"Describe"→ 准确描述黑色火车+树,无幻觉;"Is there a train?"→ `Yes` 正确。
- **坑**:legacy 脚本需 `PYTHONPATH=gguf-py`;**勿加** `--clip-model-is-vision`(CLIP config 嵌套);projector 输出维须 = LLM n_embd(1024)。详见 `docs/llamacpp_pipeline.md` mmproj 专节。
- **至此纯代码 workstream 全部对齐**;唯一未做为 #14 Xavier NX 实机(用户决定不做)。
