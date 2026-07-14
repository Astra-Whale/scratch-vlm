# AGENT_CROSSCHECK · 项目状态与交叉检查手册

**Date:** 2026-07-14
**用途:** 供其他 agent **独立交叉检查**本项目的现状与每一条硬数据。本文件强调:(a) 每个数字**从哪来、怎么复核**;(b) 易混淆点;(c) 已知坑。规划见 [`ROADMAP.md`](ROADMAP.md),历史迁移见 [`AGENT_HANDOFF.md`](AGENT_HANDOFF.md)。

---

## 0 · 交叉检查者请先读这条

**最省事的验证路径(无需 GPU / 模型 / 重新下载数据):**
仓库已提交 `logs/eval_flickr8k_test_1000.json`(1000 条 `{image, gen, refs}`)。直接:
```bash
conda activate dl
python benchmark/eval_coco_metrics.py --json logs/eval_flickr8k_test_1000.json --no-spice
```
应复现:**CIDEr 0.9268 · BLEU-4 31.73 · BLEU-1 76.16 · METEOR 0.2738 · ROUGE-L 0.5636**。
这直接验证了旗舰指标,不依赖 checkpoint 或 flickr8k 图像(两者都 gitignored)。

---

## 1 · 项目一句话

面向 **DJI 端侧 AI 系统工程师岗** 的作品级 demo。LLaVA v1.5 风格:**冻结** CLIP-ViT-L/14@336 + **冻结** Qwen2.5-0.5B-Instruct + **可训** 2 层 MLP projector(1024→2048→896,**3.94M,占总 ~801M 的 0.49%**)。卖点是端侧全流程(训练→推理→量化→迁移)+ 参数效率,不是刷 captioning 榜。

## 2 · 环境

- conda env **`dl`**:Python 3.11,**torch 2.11+cu128**,transformers 5.13,torchao 0.17,pycocoevalcap,pyarrow。
- 硬件:RTX 5060 Ti 16G(Blackwell **sm_120**)。Blackwell 需 torch ≥ 2.7 / CUDA 12.6+。
- 离线跑加 `HF_HUB_OFFLINE=1`。模型在 HF cache:CLIP-L/14@336、Qwen2.5-0.5B-Instruct、(基线用)CLIP-B/32 本地 `models/`、SmolLM2-360M。
- `conda run -n dl` 会缓冲 stdout;要实时日志加 `PYTHONUNBUFFERED=1`。

## 3 · 权威硬数据(⚠️ 注意每行来自哪个数据集/ckpt)

### 3a · 旗舰:干净可对标(Flickr8k proper split)
- **ckpt**:`checkpoints/projector_flickr8k_best.pt`(**从零训练**于 Flickr8k train 5999 图,非 init)
- **eval**:标准 Flickr8k **test 1000 图 × 5 参考**
- **官方 pycocoevalcap**:**CIDEr 0.927(×100=92.7) · BLEU-4 31.73 · BLEU-1 76.2 · METEOR 0.274 · ROUGE-L 0.564**
- 手写 corpus BLEU-4 **31.76%**(与官方 31.73 差 0.03 → 手写口径已被官方复核)
- 对标:Show-Attend-Tell Flickr8k BLEU-4 ~19.5/21.3 → **超出约 10 点**(同数据集/指标/参考数)

### 3b · 架构开发期(⚠️ Flickr30k test-1k 子集,train-on-test,已被 3a 取代作对标)
- **ckpt**:`checkpoints/projector_L14_qwenInstruct_ft_best.pt`
- **数据**:`data/flickr_1k/` 实为 **Flickr30k Karpathy test 的 1000 图**,900/100 内部切分(**训练用到了 test**)
- corpus BLEU-4 **20.59%** / sentence 22.18%(100 val)。**仅作架构演进记录,勿用于论文对标。**

### 3c · 量化(⚠️ 在 3b 的 arch-dev ckpt + Flickr30k-test-100 上测,非 Flickr8k)
| quant | corpus BLEU-4 | 权重常驻显存 |
|---|---|---|
| bf16 | 20.59% | 1536 MB |
| int8(排除 lm_head) | 20.04% | 1191 MB (-22%) |
| int4(排除 lm_head,tinygemm) | 18.33% | 1057 MB (-31%) |
- 混合精度:只压 Qwen transformer Linear,CLIP/projector/lm_head 守 bf16。详见 [`docs/quantization_plan.md`](docs/quantization_plan.md)。

### 3d · latency / 迁移(arch-dev ckpt)
- 推理 batch=1:**1.6 GB**;视觉编码 15.3ms(一次性)+ prefill 14.1ms + decode 7.29ms/tok = **137 tok/s**(`logs/latency_profile.json`)。
- 迁移:5060 Ti 训练 300 步 **28.2s / VRAM 2.63GB** vs 4060 5.8min/2.67GB;4060 ckpt 跨平台复评 sentence **16.98%**(≈原 17.15%)。

## 4 · 如何复核每个数字

| 声明 | 复核命令 | 需要 |
|---|---|---|
| 3a 官方指标 | `python benchmark/eval_coco_metrics.py --json logs/eval_flickr8k_test_1000.json --no-spice` | 仅提交的 JSON(**无需模型/数据**) |
| 3a 手写 BLEU | `python evaluate.py --ckpt checkpoints/projector_flickr8k_best.pt --data data/flickr8k/test.jsonl --image-root data/flickr8k/images --max-samples 1000` | ckpt + flickr8k 数据(均 gitignored,需重建) |
| 3c 量化 | `python evaluate.py --ckpt checkpoints/projector_L14_qwenInstruct_ft_best.pt --quant int8 --max-samples 100` | ckpt + flickr_1k 数据 |
| 3d latency | `python benchmark/profile_latency.py` | ckpt |
| 数据重建(flickr8k) | 下 `jxie/flickr8k` 4 个 parquet 到 `data/flickr8k/` → `python data/prepare_flickr8k.py` | ~1.1GB 下载 |

> **注**:`checkpoints/*.pt` 与 `data/` blob 均 **gitignored**(未进仓库)。要跑需模型的复核,须先重建数据(§4 末行)+ 重训,或索取 ckpt。**无模型的复核走 §0。**

## 5 · 文件地图

**代码**:`model/`(vlm/vision_encoder/projector)· `train.py`(--grad-accum / --init-projector)· `evaluate.py`(corpus+sentence BLEU / --quant int8|int4 / 路径 fallback)· `inference.py` · `app.py`(Gradio)· `data/*.py`(prepare_flickr8k 等)· `benchmark/`(profile_latency / export_onnx / eval_coco_metrics)
**文档**:`ROADMAP.md` · `docs/`(quantization_plan / benchmark_landscape / competitor_benchmark / talkshop_qa / pitch / cv_entry / s2_aerial_plan)· `benchmark/migration_analysis.md`
**证据**:`logs/`(训练日志 + 各 eval JSON,已提交)
**gitignored**:`checkpoints/`(权重)· `data/` blob · `models/`(缓存)· `onnx/`

## 6 · 易混淆点(交叉检查重点核对)

1. **两套数据别混**:Flickr8k(3a,干净,旗舰)vs Flickr30k-test-1k(3b,train-on-test,旧)。`data/flickr_1k/` 名字叫 1k 但**是 Flickr30k 的 test split**。
2. **量化/latency 数字在 arch-dev ckpt(3b/3c/3d)上测,不是 Flickr8k**。若要 Flickr8k 上的量化数需另跑。
3. **CIDEr 跨数据集不可比**:Flickr8k 的 92.7 ≠ COCO SOTA 130-155(不同语料 TF-IDF)。
4. **BLEU 两种口径**:corpus(标准、可对标)vs sentence-level+平滑(内部追踪,偏高)。
5. **手写 BLEU 已被官方 pycocoevalcap 复核**(31.76 vs 31.73)。

## 7 · 已知坑(踩过并修复,供核对代码)

1. **Qwen base ≠ Instruct**:base 冻结 LM 不吐 ChatML `<|im_end|>` → 生成越过 caption 吐垃圾 → BLEU 22%→12%。必须用 **Instruct**;`model/vlm.py::generate` 已把 `<|im_end|>` 加进 eos。
2. **torchao int4 默认 packing 需 `mslk`(非便携,撞 Orin 红线)** → 用 `int4_packing_format="tile_packed_to_4d"`(tinygemm,Ampere 原生)。见 `evaluate.py`。
3. **量化 tied lm_head 反效果**:Qwen `tie_word_embeddings=True`,量化 lm_head 打破共享(+136MB)且伤精度 → `filter_fn` 排除 lm_head。
4. **完整 Flickr30k(nlphuji)zip 走 xethub CDN 本环境限流下不动** → 换 `jxie/flickr8k`(parquet 普通 CDN)。
5. **fp16 + 随机 projector 在 test_forward 偶发 loss=NaN**(cosmetic);bf16 稳定,项目全程 bf16。
6. **跨平台 ckpt 路径**:旧 ckpt 存 Windows 绝对路径;`evaluate.py`/`inference.py` 仅对**绝对/盘符路径**做本地 fallback(HF repo_id 不碰)。

## 8 · 未完成 / 暂缓 / 待人工

- **S2 航拍领域适配**:用户暂缓,spec 见 `docs/s2_aerial_plan.md`(一条命令可执行,差数据集)。
- **git 身份是占位** `徐悦 <xuyue@localhost>`(本地配置),需改真实邮箱。
- **SOTA ckpt gitignored**(`*.pt`);若要开箱可跑可 `git add -f checkpoints/projector_flickr8k_best.pt`(7.5MB)。
- **无 git remote**;未推送。
- **SPICE 未跑**(需大内存 Java、1000 图易卡);CIDEr 已是主指标。
- **COCO 同轴对标未做**(需 COCO 训练,大成本且注定不 competes SOTA;结论:不比数字,比范式+效率,见 `docs/benchmark_landscape.md`)。

## 9 · 记忆锚点

Agent 记忆库(`.claude/.../memory/`)有 `vlm-strategy-constraints`(8GB 上限 / Orin 红线 / SOTA / 量化要点)与 `vlm-migration-findings`。与本文一致;若冲突以本文 + 仓库实测为准。
