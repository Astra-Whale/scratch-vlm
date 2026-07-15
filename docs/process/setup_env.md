# 环境设置详情

本项目在两台硬件上验证过跨平台一致性。**当前开发平台是 5060 Ti (Ubuntu)**;
4060 (Windows) 为前期 baseline,保留作历史记录。

## 当前平台 · RTX 5060 Ti (Ubuntu)

### 硬件
- GPU: NVIDIA RTX 5060 Ti · **16 GB VRAM** · Blackwell (sm_120) · driver 580.x
- OS: Ubuntu (Linux)

### conda env `dl`
| 组件 | 版本 |
|------|------|
| Python | 3.11 |
| torch | **2.11+cu128** (Blackwell 需 torch ≥ 2.7 / CUDA 12.6+) |
| transformers | 5.13.x |
| accelerate / sentencepiece / pillow / huggingface_hub | 最新 |

```bash
conda activate dl
pip install transformers accelerate sentencepiece pillow huggingface_hub
# 若自建: conda create -n vlm python=3.10 && pip install torch --index-url https://download.pytorch.org/whl/cu126
```

`torch.cuda.get_device_capability(0) == (12, 0)`,`torch.cuda.is_bf16_supported() == True`。

### VRAM 实测 (bf16, CLIP-L/14@336 + Qwen2.5-0.5B-Instruct)
| 场景 | 显存峰值 | 备注 |
|------|---------|------|
| **推理 batch=1** | **1.6 GB** | 端侧可部署性关键指标; 稳进 8GB / Orin Nano 8G |
| 训练 batch=4 × grad-accum=4 | ~9.9 GB | 576 visual tokens 较重; batch=8 连 16G 都 OOM |
| 纯权重常驻 | ~1.5 GB | |

> 16G 仅用于提升训练迭代效率(更大 micro-batch / grad-accum),**交付模型推理锁定 8GB 内**。

## 前期平台 · RTX 4060 Laptop (Windows, 历史)

- GPU: RTX 4060 Laptop · 8 GB · Ada Lovelace (sm_89);OS: Windows;env `cu12` (Python 3.10, torch 2.6.0+cu124)
- 前期架构 CLIP-B/32 + SmolLM2-360M,训练峰值 ~2.67 GB (batch=8)
- 同一份代码零改动迁到 5060 Ti;迁移一致性见 `AGENT_HANDOFF.md` §5 与 `benchmark/migration_analysis.md`

## HuggingFace 缓存

Linux: `~/.cache/huggingface/hub/`。首次需下载(精确下法避免拉全 repo):
```bash
hf download openai/clip-vit-large-patch14-336 pytorch_model.bin
hf download openai/clip-vit-large-patch14-336 config.json preprocessor_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt
hf download Qwen/Qwen2.5-0.5B-Instruct model.safetensors
hf download Qwen/Qwen2.5-0.5B-Instruct config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt
```
