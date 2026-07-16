# scratch-vlm

从零拼装并训练的轻量视觉-语言模型:**冻结 CLIP-ViT-L/14 视觉塔 + 2 层 MLP projector + 冻结 Qwen3-0.6B**,LLaVA v1.5 风格。覆盖两阶段训练、captioning 与 POPE 评测、llama.cpp 端侧量化推理(GGUF Q4_K_M + mmproj)。

面向边缘部署(Jetson-class)设计,交付模型推理显存在 8GB 内。开发环境 RTX 5060 Ti 16G / Ubuntu。

## 架构

```
图像 → CLIP-ViT-L/14@336 (冻结) → [B, 576, 1024]
                                      ↓
                        2 层 MLP Projector (唯一可训, 4.20M)
                                      ↓
                              [B, 576, 1024]
                                      ↓
文本 (含 <image> 占位) ──→ 拼接 [文本] [视觉 tokens] [文本] ──→ Qwen3-0.6B (冻结) → 文本
```

`<image>` token 在 forward/generate 时被替换为投影后的视觉 tokens,与文本 embedding 在序列维拼接后送入 LLM。

| 组件 | 参数量 | 状态 |
|-----|--------|------|
| CLIP-ViT-L/14@336 视觉塔 | ~304M | 冻结 |
| Qwen3-0.6B LLM | ~596M | 冻结 |
| MLP Projector (1024→2048→1024) | **4.20M** | 唯一可训 |
| 总 / 可训 | ~904M / 4.20M | 0.46% |

## 训练

两阶段,CLIP 与 Qwen3 主干全程冻结。

| 阶段 | 可训部分 | 数据 | 配置 |
|-----|---------|------|------|
| **stage-1** projector 对齐 | projector (4.20M) | Flickr8k train (5999 图 × 5 caption = 30000 pair) | batch 4 / grad-accum 4 / lr 1e-3 / bf16 |
| **stage-2** 指令微调 | projector + Qwen q/v LoRA (r=16, α=32) | LLaVA-Instruct (detail_23k + conversation_58k) + VQAv2 平衡 yes/no,共 11022 条 | batch 1 / grad-accum 16 / 700 步 / lr 2e-4 |

stage-2 可训参数 6.49M(LoRA 2.29M + projector 4.20M),峰值显存 7.2GB。

## 结果

### Image Captioning · Flickr8k 标准 split

stage-1 模型在 Flickr8k 1000-test(5 参考)上,官方 `pycocoevalcap`:

| CIDEr | BLEU-4 | BLEU-1 | BLEU-2 | BLEU-3 | METEOR | ROUGE-L |
|-------|--------|--------|--------|--------|--------|---------|
| **0.940** | **32.91** | 75.8 | 59.6 | 44.9 | 0.276 | 0.573 |

（本项目手写 corpus BLEU-4 = 32.93,与官方一致。）

> BLEU-1..4 为 pycocoevalcap 累积口径(几何平均,与论文对标)。SPICE 因 Java 内存约束未跑,CIDEr 为主对标指标。

### POPE 幻觉评测 · stage-2 模型

物体存在性 yes/no,COCO val2014,三 split 各 3000 题:

| split | acc | precision | recall | f1 | yes 占比 |
|-------|-----|-----------|--------|-----|---------|
| random | 82.9 | 81.1 | 85.7 | 83.4 | 52.8 |
| popular | 76.1 | 71.9 | 85.7 | 78.2 | 59.6 |
| adversarial | 70.2 | 65.4 | 85.7 | 74.2 | 65.5 |

平均 F1 **78.59**。

SFT 数据配比对 POPE 的影响(消融):

| SFT 数据 | acc | avg F1 |
|---------|-----|--------|
| detail(纯描述) | 50 | 0 |
| + conversation | 50 | 66.7 |
| + 平衡 VQAv2 | 70–83 | **78.59** |

> 三行均 n=3000(三 split 各 3000 题)。detail-only F1=0 不是"低幻觉":该模型只输出描述文本、不会短答 yes/no(**unparseable=100%**),parser 保守判 no → 全 no → acc≈50(yes/no 各半的随机水平)、F1=0。+conversation 学会答但偏 yes(训练短答 93% 是 yes),F1 66.7 虚高;+平衡 VQAv2 才均衡。

### 量化 · llama.cpp

Qwen3-0.6B 转 GGUF:

| 格式 | 体积 | PPL (wikitext) |
|-----|------|----------------|
| f16 | 1.5 GB | 19.63 |
| **Q4_K_M** | **0.48 GB**(3.12×) | 21.35 |

> PPL 语料:wikitext-2 test 前 90KB / 43 × 512-token chunks / ctx=512(小切片,关注 f16→Q4 的相对退化 +8.7%)。另有 fp16 vs Q4_K_M 的 VQA 定性对照(`logs/vqa_fp16_vs_q4.json`)。

llama-server SSE 流式输出跑通;详见 [`docs/llamacpp_pipeline.md`](docs/llamacpp_pipeline.md)。

### 端侧多模态推理 · mmproj

CLIP + projector 打包为 llama.cpp mmproj GGUF(590MB),与合并 LoRA 后的 Qwen3 Q4_K_M(372MiB)组合,`llama-mtmd-cli` 端到端图文推理跑通(图像描述 + yes/no 问答)。

### 推理开销

5060 Ti · bf16 · batch=1 · **Qwen3-0.6B + CLIP-L/14@336**(stage-1 旗舰 ckpt)实测:视觉编码 14.8ms(一次性)+ prefill 20.7ms + decode **94 tok/s**(memory-bandwidth-bound)。视觉编码仅占 32-token 生成的 4.1%。`logs/latency_profile.json` / `benchmark/profile_latency.py`。

> decode/token 用 (g64−g16)/48 差分法,假设 decode 恒定、忽略 KV-cache 增长的 O(n) 项(对分档估计够用)。

## 目录结构

```
scratch-vlm/
├── train.py               # stage-1 projector 训练
├── train_sft.py           # stage-2 LoRA + projector SFT
├── evaluate.py            # captioning BLEU 评测 (corpus + sentence)
├── inference.py           # 单图推理
├── app.py                 # Gradio demo
├── model/                 # vision_encoder(CLIP) · projector(MLP) · vlm(拼装+forward+generate)
├── data/                  # dataset · sft_dataset · prepare_flickr8k · fetch_coco_images
├── benchmark/             # eval_coco_metrics · evaluate_pope · profile_latency · deploy_orin_nx_8g.md
├── tools/                 # merge_lora (合并 LoRA→导 GGUF 前置) · build_llama_orin.sh
├── tests/                 # test_forward
├── docs/                  # 技术笔记 (llamacpp_pipeline · benchmark_landscape · data_sourcing)
├── logs/                  # 评测/训练 evidence (POPE / Flickr8k / PPL / latency 的 JSON+日志)
├── checkpoints/           # 权重 (gitignored;旗舰 projector_stage1_qwen3_best.pt 已入库)
└── weights/               # HF/GGUF/合并模型缓存 (gitignored)
```

## 环境

conda env `dl`:Python 3.11 · torch 2.11+cu128(RTX 5060 Ti / Blackwell sm_120,需 torch ≥ 2.7)。

```bash
conda activate dl
pip install -r requirements.txt
```

首次运行自动从 HuggingFace 下载 CLIP-ViT-L/14@336 (~1.7GB) 与 Qwen3-0.6B (~1.2GB)。

## 快速开始

```bash
# 前向验证 (参数量 / shape / VRAM)
python tests/test_forward.py

# stage-1 · projector 对齐 (Flickr8k)
python train.py --data data/flickr8k/train.jsonl --val-data data/flickr8k/val.jsonl \
                --image-root data/flickr8k/images \
                --vision openai/clip-vit-large-patch14-336 --llm weights/Qwen3-0.6B \
                --steps 3000 --batch 4 --grad-accum 4 --lr 1e-3 --dtype bf16 \
                --out checkpoints/projector_stage1_qwen3.pt

# stage-2 · LoRA + projector SFT
python train_sft.py --data data/llava_instruct/sft_mix2.json --image-root data/coco/train2014 \
                --vision openai/clip-vit-large-patch14-336 --llm weights/Qwen3-0.6B \
                --init-projector checkpoints/projector_stage1_qwen3_best.pt \
                --steps 700 --batch 1 --grad-accum 16 --lr 2e-4 --lora-rank 16 \
                --out checkpoints/vlm_stage2_mix2

# captioning 评测
python evaluate.py --ckpt checkpoints/projector_stage1_qwen3_best.pt \
                --data data/flickr8k/test.jsonl --image-root data/flickr8k/images --max-samples 1000

# 官方指标 (无需 GPU/模型, 用已提交的 eval JSON)
python benchmark/eval_coco_metrics.py --json logs/eval_flickr8k_qwen3_test_1000.json --no-spice

# POPE 幻觉评测
python benchmark/evaluate_pope.py --projector-ckpt checkpoints/vlm_stage2_mix2/projector.pt \
                --lora-adapter checkpoints/vlm_stage2_mix2/lora_adapter \
                --pope-dir data/pope --image-root data/coco/val2014

# 端侧多模态推理 (llama.cpp)
llama-mtmd-cli -m weights/gguf/qwen3-stage2-merged-q4_k_m.gguf \
               --mmproj weights/gguf/mmproj-model-f16.gguf --image <img>.jpg -p "Describe this image."
```

## 权重与数据

数据集、HF/GGUF 缓存、LoRA adapter 均 gitignored,仓库只含代码、文档与评测 JSON。旗舰 projector `checkpoints/projector_stage1_qwen3_best.pt`(8.4MB)已入库,下载 CLIP-L + Qwen3 后即可跑 captioning 评测。其余权重按上方命令重建。
