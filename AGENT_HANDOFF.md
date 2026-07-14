# Agent 交接文档 · scratch-vlm 项目

**Date:** 2026-07-13
**给谁看:** 在 Ubuntu / RTX 5060 Ti 16G 上接手本项目的 Claude Code Agent
**你的第一件事:** 读完本文件 → 按 §5 完成迁移验证 → 汇报结果给用户
**前置阅读:** 本文件后可选读 [`MIGRATE_TO_UBUNTU.md`](MIGRATE_TO_UBUNTU.md)（详细命令 checklist）+ [`README.md`](README.md)（项目门面）+ [`benchmark/migration_analysis.md`](benchmark/migration_analysis.md)（硬件对比）

---

## 0 · 你是谁 · 你的任务

**你**：Ubuntu 上的 Claude Code Agent，刚接手一个从 Windows/RTX 4060 迁过来的 VLM 项目。用户是徐悦（2027 届 ShanghaiTech CS 应届生），本项目是他为 **DJI 端侧 AI 系统工程师岗（提前批, 报告评分 4.8/5）**打造的作品级演示。

**你的核心任务**：
1. **§5**：先在 Ubuntu 5060 Ti 上把项目**跑通**（3 项验证：test_forward / inference / evaluate）
2. **§6**：跑通后按用户指示推进"未完成待办" (P0-P3)
3. **§7**：与用户协作时的约定

**你的操作权限**（默认）：
- ✅ 建 conda env、装 pip 包
- ✅ 跑 shell 命令
- ✅ 读写项目目录 `~/vlm/`（假设用户已放在这）
- ⚠️ 大文件下载（>500MB）前先跟用户确认（预估时间和硬盘占用）
- ⚠️ 涉及删除 checkpoint / logs 前先确认

---

## 1 · 项目背景与叙事

### 目标岗位

**DJI 端侧 AI 系统工程师** (2027 提前批, 上海) · [reports/015-dji-edge-ai-2026-07-01.md](../27提前批和秋招/career-ops-试跑/reports/015-dji-edge-ai-2026-07-01.md)

**JD 关键条款**（决定本项目的所有技术选择）:
- 深度学习编译器 · 混合精度量化 · 异构调度运行时 · 算子性能调优 · 端侧大模型
- 加分项: 机载机器人 AI 部署经验 · RoboMaster/ROBOCON 经验（用户是 RM 三年队长, 钻石加持）

### 项目定位

**"散装 VLM"** = 参考 LLaVA v1.5 架构，自己拼装 CLIP + MLP-Projector + LLM 的极简 VLM，跑在端侧硬件（4060 baseline → 5060 Ti 升级 → 未来 Jetson-class）上，重点展示：
- **端侧 AI 全流程能力**（训练 → 推理 → 量化 → 功耗 profile → 硬件迁移分析）
- **仅训 3.5M projector，冻结 vision/LLM 主干** 的经济学训练策略
- **面向 Ampere+ Jetson-class 硬件设计**，bf16 权重可零转换迁移

### 硬件切换的意义

- **4060 Laptop 8G (Windows, Ada Lovelace sm_89)**: 前期 baseline，pipeline 已跑通
- **5060 Ti 16G (Ubuntu, Blackwell sm_120)**: **你现在的环境**，为升级实验（CLIP-L/14, Qwen2.5, 长 epoch, fp8）而来

**面试叙事**（你要理解，因为面试官会追问）:
> "我在两台硬件上验证 pipeline 跨平台性——4060 Laptop 作为 baseline，5060 Ti Desktop Ubuntu 作为升级平台。同一份代码零改动跨两平台跑通。5060 Ti 上进一步利用 Blackwell 5th-gen Tensor Core 的 fp8/fp4 特性做端侧量化实验。"

---

## 2 · 现状快照（Windows 侧完成的东西）

### 已完成里程碑（按时间顺序）

| 阶段 | 内容 | 硬数据 |
|------|------|-------|
| **D1** | Skeleton + 前向验证 | 452.8M 总参 / 3.54M 可训 (0.78%) · VRAM 峰值 1.1 GB (fp16, batch=2) |
| **D2** | 训练闭环 + 推理 (toy 24 imgs) | 60 步 loss **-77%** (3.57→0.82) · 197ms/step · bf16 · batch=4 |
| **D2.5** | Flickr1K 真数据 pretrain (300 步) | 5000 pairs, loss **-24%** (3.63→2.76) · 1165ms/step · batch=8 · VRAM 2.67 GB |
| **D2.6** | Train/Val 划分 + 手写 BLEU-4 + Baseline 对比 | **Baseline 3.15% → Trained 17.15% (+5.4×)** on 100 val 图 · greedy decoding |
| **D2.7** | 4060/Xavier/Orin 迁移分析文档 | `benchmark/migration_analysis.md` 15KB |

### 面试可讲的核心硬数据

| 类别 | 数据 |
|------|------|
| **架构** | LLaVA v1.5 风格 · CLIP-B/32 + MLP(3.5M) + SmolLM2-360M-Instruct · 452M 总/3.54M 可训 |
| **训练** | Flickr1K 5K pairs · 900/100 image split · bf16 · AdamW · cosine schedule · Chen&Cherry BLEU 手写 |
| **性能** | 300 步 5.8min · 1165ms/step · VRAM 峰值 2.67 GB · loss -24% |
| **评测** | BLEU-4 **17.15%** (trained) vs **3.15%** (baseline) = **+5.4×** on 100 val 图 |
| **迁移分析** | 4060 → Orin NX/Nano bf16 零转换 · 内存带宽 272→102 GB/s |

### 关键 checkpoint

- `checkpoints/projector_toy.pt` — D2 toy 训 60 步（结果不重要，只是 pipeline demo）
- `checkpoints/projector_flickr1k.pt` — D2.5 Flickr 300 步（**目前 SOTA**, BLEU 17.15%）
- `checkpoints/projector_flickr1k_best.pt` — D2.5 val loss 最优点（若存在）

---

## 3 · 项目文件结构 & 每个文件的角色

```
vlm/
├── README.md                     ← 项目门面, 全部硬数据
├── setup_env.md                  ← 环境说明（Windows 视角, 你可以更新为 Ubuntu）
├── MIGRATE_TO_UBUNTU.md          ← 迁移到 Ubuntu 的详细 checklist（原始指令）
├── AGENT_HANDOFF.md              ← 本文件
├── requirements.txt              ← 依赖清单（版本未 pin）
├── .gitignore                    ← 忽略 data/ checkpoints/ models/ 等大文件

├── model/                        ← 三段模型拼装
│   ├── __init__.py
│   ├── vision_encoder.py         ← CLIP-ViT 冻结版, select_layer=-2 (LLaVA 官方)
│   ├── projector.py              ← 2 层 MLP (input=1024/CLIP-L or 768/CLIP-B, output=llm_hidden)
│   └── vlm.py                    ← 拼装 + forward(training) + generate(inference) + <image> token 替换逻辑

├── data/
│   ├── toy_dataset.py            ← 生成 24 张合成图 (PIL, 8色 x 3形状)
│   ├── dataset.py                ← MVPCaptionDataset + ChatML prompt wrapping + label masking
│   ├── prepare_flickr.py         ← 解压 Flickr zip + CSV→JSONL, 已过滤 macOS metadata
│   ├── split_flickr.py           ← 900/100 image-level split (防 caption leak)
│   ├── toy.jsonl · toy_images/   ← D1 合成数据
│   └── flickr_1k/                ← 真数据 (200 MB)
│       ├── test_1k_flickr.csv    ← 原始 1000 图 x 5 caption
│       ├── flickr_1k.jsonl       ← 全 5000 pairs 展平版
│       ├── flickr_1k_train.jsonl ← 4500 pairs (900 imgs)
│       ├── flickr_1k_val.jsonl   ← 500 pairs (100 imgs)
│       └── images/*.jpg          ← 1000 张真 Flickr 图

├── tests/test_forward.py         ← D1 验证脚本 · 参数量 + shape + VRAM + loss 一次性
├── train.py                      ← CLI 训练脚本 (支持 --val-data + best-ckpt saving + LR schedule)
├── inference.py                  ← 单图推理 (支持 --baseline / --ckpt / --no-load)
├── evaluate.py                   ← 手写 BLEU-4 (multi-ref, Chen&Cherry smoothing) + baseline 对比

├── benchmark/
│   └── migration_analysis.md     ← 4060/Xavier/Orin 三代对比 + 面试 talk shop 4 问答

├── checkpoints/
│   ├── projector_toy.pt
│   ├── projector_flickr1k.pt     ← 目前 SOTA
│   └── projector_flickr1k_best.pt (可能存在)

├── logs/
│   ├── train_run_*.log
│   ├── eval_baseline_100.json    ← 100 张 val 详细结果
│   └── eval_trained_100.json

└── models/models/openai-mirror--clip-vit-base-patch32/
    └── snapshots/master/          ← 605MB 本地 CLIP-B/32（用户已复制过来, 无需重新下）
        ├── pytorch_model.bin (605MB)
        └── config.json / preprocessor_config.json / tokenizer* / ...
```

**HuggingFace cache**: `~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct/` — Ubuntu 上用户可能已复制过来。若没有，首次跑 test_forward.py 会 auto-download（720MB, 通过代理约 20-30 分钟）。

---

## 4 · 关键技术决策与踩过的坑

### 4.1 已锁定的技术决策（不要动）

| 决策 | 理由 | 面试话术 |
|-----|------|---------|
| **LLaVA v1.5 风格 (MLP > Q-Former)** | 论文消融证明 | "MLP 参数少 10x, 精度相当或更优" |
| **冻结 CLIP + LLM, 只训 Projector** | 端侧经济学最优 | "3.5M 可训 = 452M 总参的 0.78%" |
| **CLIP `select_layer=-2` (倒数第二层)** | LLaVA v1.5 官方 | "最后一层过于向语言塌缩, 失去视觉细节" |
| **bf16 训练（非 fp16）** | 数值稳定性 + Ampere+ 原生支持 | "bf16 有 fp32 同样的动态范围, 避免 loss underflow" |
| **ChatML 格式 wrapping prompt** | SmolLM2-Instruct 必需, 裸 prompt 会秒 EOS | "Instruct 模型必须走 chat template, 否则 EOS 短路" |
| **手写 BLEU-4 (Chen&Cherry smoothing)** | 免第三方依赖 + 面试可讲深度 | "modified n-gram + brevity penalty + method-1 smoothing" |
| **按 image 层面划分 train/val** | 防 caption leak | "5 条 caption 属于同一张图, 按 caption 划会 leak" |

### 4.2 已踩过的坑（避免重踩）

**坑 1**：huggingface_hub v1.23+ 的 xet 协议与 hf-mirror 不兼容
- 症状：`FileMetadataError: Distant resource does not seem to be on huggingface.co`
- Windows 侧 workaround：走 modelscope + curl 直下
- **Ubuntu 侧新情况**：用户装了代理，直连 huggingface.co 应该能过 HfApi。但下载大文件时可能仍慢（331 KB/s），必要时仍走 curl+modelscope

**坑 2**：CLIPVisionModel 加载完整 CLIP 权重时报 UNEXPECTED text_model 层
- 症状：一堆 `text_model.encoder.layers.{0..11}...  | UNEXPECTED` 警告
- 判断：**warning 不影响功能**，因为完整 CLIP 权重里含 text_model 但我们只用 vision，PyTorch 自动跳过
- 处理：忽略即可，无需 hack

**坑 3**：Windows GBK 控制台无法输出 unicode 符号（✓ ✗ 等）
- 症状：`UnicodeEncodeError: 'gbk' codec can't encode character '✓'`
- 已修：test_forward.py / train.py / inference.py / evaluate.py 都加了 `sys.stdout.reconfigure(encoding='utf-8')`
- Ubuntu 上默认 UTF-8，此段代码是 no-op，无害

**坑 4**：Windows 反斜杠路径被 transformers 当 repo_id
- 症状：`HFValidationError: Repo id must use alphanumeric chars...`
- 已修：所有本地路径用 `Path(...).as_posix()` 转正斜杠
- Ubuntu 上无此问题，但代码保留即可（跨平台）

**坑 5**：modelscope 下载时会拉多种权重格式（pytorch/flax/tf）
- 症状：目录里出现 `flax_model.msgpack.incomplete` `tf_model.h5.incomplete`
- 已知：调 modelscope.snapshot_download 无法只下单一格式（除非用低层 API 或用 curl）
- Ubuntu 侧策略：**若用 huggingface-cli download 更精准可控**（能只下 pytorch_model.bin 而不下 flax）

**坑 6 (Blackwell 特有)**：PyTorch 2.6 不支持 sm_120
- 症状：`RuntimeError: CUDA error: no kernel image is available for execution on the device`
- **修复：Ubuntu 上必须装 PyTorch >= 2.7 with CUDA 12.6+**
- 命令：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`

### 4.3 待验证的假设

- **假设 A**：5060 Ti 16G 上跑 CLIP-B/32 + SmolLM2 batch=8 训练 VRAM ~2G（对比 4060 上 2.67G）
- **假设 B**：5060 Ti 上 300 步 batch=8 训练 wall time ~3-4 分钟（对比 4060 的 5.8 分钟）
- **假设 C**：迁移到 Ubuntu 后 loss/BLEU 数字应与 Windows 一致（同 seed, 同 dtype）
- **假设 D**：Blackwell fp8 tensor core 通过 transformer-engine 库可访问

---

## 5 · 你的首要任务：迁移验证（3 步）

**目标**：在 5060 Ti + Ubuntu 上把现有 pipeline 跑通，确保迁移一致性。

### 5.1 · 环境搭建（15-30 分钟）

前置：Ubuntu 已装、NVIDIA driver 已装、nvidia-smi 能显示 5060 Ti

```bash
# 1. Miniconda（若没）
if ! command -v conda &> /dev/null; then
    cd /tmp && wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    $HOME/miniconda3/bin/conda init bash
    source ~/.bashrc
fi

# 2. 建 env
conda create -n vlm python=3.10 -y
conda activate vlm

# 3. 装 PyTorch 2.7+ (Blackwell 支持)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 4. 装项目依赖
pip install transformers accelerate sentencepiece pillow modelscope

# 5. 验证 CUDA + Blackwell
python -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0))
print('compute cap:', torch.cuda.get_device_capability(0))  # 期望 (12, 0)
print('VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
print('bf16:', torch.cuda.is_bf16_supported())
"
```

**期望**：`compute cap: (12, 0)` (Blackwell sm_120), VRAM 15.9-16.0 GB, bf16 True。**若 `cuda: False` 或 `compute cap` 不是 12,x → 停下问用户 driver/CUDA 版本，别硬跑**。

### 5.2 · 前向验证（5 分钟）

```bash
cd ~/vlm  # 或者用户放的路径

# 若 HF cache 里有 SmolLM2 (用户复制过来了), 直接跑; 若没, 首次会 auto-download
python tests/test_forward.py
```

**期望输出**：
```
[env] device=cuda, dtype=torch.float16
      GPU: NVIDIA GeForce RTX 5060 Ti
      VRAM total: 16.0 GB
[1] 加载 VLM (vision=<local path>, llm=HuggingFaceTB/SmolLM2-360M-Instruct)...
    ✓ 加载完成 (耗时 x s)
[2] 参数量统计
    可训参数量: 3.54M  (与 4060 一致)
[3-4] forward 通过, loss 有值
[5] VRAM 峰值: 1000-1500 MB (5060 Ti 上可能比 4060 略低, 因为架构效率更高)
✓ Week 1 D1 里程碑达成
```

**若失败**：贴 error 给用户，别自己乱改代码。

### 5.3 · 推理 + 评测一致性验证（10 分钟）

```bash
# 用现有 checkpoint 跑一张 toy 图, 验证输出与 Windows 一致
python inference.py --image data/toy_images/000_orange_square.png --dtype bf16 --temperature 0.0

# 期望: "A green triangle." 或类似 (D2 训 toy 时的输出, seed 一致会一致)

# 用 Flickr checkpoint 跑 20 张 val, 看 BLEU
python evaluate.py --ckpt checkpoints/projector_flickr1k.pt --max-samples 20 --show-samples 3
```

**期望**：BLEU-4 在 **0.13-0.15** 范围（20 张 sample 有随机性，100 张是 0.1715）。**若差异 > 20%**，说明模型加载有问题，报告给用户。

### 5.4 · 迁移完成的对比表（更新 README）

跑完后建议**主动**在 README 里加一张对比表：

| 测试 | 4060 Laptop (Windows) | 5060 Ti (Ubuntu) | 提升 |
|-----|----------------------|-------------------|-----|
| test_forward VRAM | 1.10 GB | ? | ? |
| Flickr 300 步训练 wall time | 5.8 分钟 | ? (跑一次填) | ? |
| 100 张 val BLEU-4 | 17.15% | ? (数字应一致) | 一致性验证 ✓ |

---

## 6 · 后续待办清单（P0-P3）

**P0 = 首要任务（§5 迁移验证已经是 P0，跑完这个才能进 P1）**

### P1 · 快速见效实验（几十分钟到 2 小时）

- **P1.1 · Flickr1K 5 epoch batch=16 训练**（20-30 分钟, 无下载）
  - 命令：`python train.py --data data/flickr_1k/flickr_1k_train.jsonl --val-data data/flickr_1k/flickr_1k_val.jsonl --image-root data/flickr_1k/images --steps 1500 --batch 16 --lr 1e-3 --dtype bf16 --val-every 50 --out checkpoints/projector_flickr1k_5ep.pt`
  - 预期：BLEU **17% → 22-25%**
  - 之后跑 `evaluate.py --ckpt checkpoints/projector_flickr1k_5ep.pt --max-samples 100`

- **P1.2 · CLIP-B/32 → CLIP-L/14 @ 336px 升级**（1-1.5 小时含下载）
  - 下载：`huggingface-cli download openai/clip-vit-large-patch14-336` 或 modelscope
  - 重训（batch 可能要降至 8-12）：`python train.py --vision openai/clip-vit-large-patch14-336 ...`
  - 预期：BLEU **+3-5%** 相对 L/14 baseline

- **P1.3 · pynvml 采 5060 Ti 功耗 profile**（30-60 分钟）
  - `pip install pynvml`
  - 写个小 script 在 train.py 或 inference.py 边跑边采
  - 输出 `benchmark/power_5060ti.md`

### P2 · 深入实验（2-6 小时）

- **P2.1 · LLM 换 Qwen2.5-0.5B**（30-60 分钟, LLaVA 官方选型）
  - 注意验证 Qwen ChatML 与 dataset.py 的 `<|im_start|>` 兼容性
- **P2.2 · LLM 升级 Qwen2.5-1.5B**（5060 Ti 16G 才能容, 4060 上跑不了）
- **P2.3 · GGUF Q4 量化 + llama.cpp benchmark**（2-4 小时, 原 D3）
- **P2.4 · Blackwell fp8 实验 via transformer-engine**（Blackwell 独有特性, 面试杀器）

### P3 · 文档/演示类

- **P3.1 · Gradio demo**（1-2 小时, 面试实况演示用）
- **P3.2 · talk shop Q/A 20 问文档**（1-2 小时, 面试前必备）
- **P3.3 · 15/2/0.5 分钟三档 pitch 稿**（30 分钟）
- **P3.4 · README 里加 4060 vs 5060 Ti 对比表**（迁移完顺手做）

**优先级建议给用户的组合**：
- **最快见效**：P1.1 (5 epoch) + P1.3 (功耗) → 1.5 小时
- **最全面**：P1.1 + P1.2 (CLIP-L) + P2.3 (量化) → 5-8 小时
- **面试冲刺**：P1.1 + P3.2 (Q/A) + P3.1 (demo) → 3-4 小时

---

## 7 · 与用户 & 前任 Agent 的协作约定

### 用户偏好（Windows 侧对话总结）

用户是徐悦，本项目主人。历史沟通模式：
- **喜欢"总分结构"**：先给结论、要点，再展开
- **反感 push**：给选项让他选，不要"我建议你必须做 X"
- **喜欢"先上车后补票"**：cv 或 README 允许写"待补数据"，跑出来再填
- **有诚信底线**：绝对不能在数据上撒谎（4060 数据不能标 Xavier NX）
- **偏"深度 > 广度"**：3+ 周只做 1-2 个方向，不铺开
- **喜欢直白诊断**：模型效果不好就说"没通", 别粉饰太平
- **中英混排**：默认中文交流，代码/文件名/技术术语英文

### 前任 agent (Windows 侧) 的做事习惯

- 用 `AskUserQuestion` 提选择题，选项 2-4 个，每个 1-2 行描述
- 长时间任务用 `Bash run_in_background=true` + `Monitor`
- 决策链尽量透明：告知假设 → 提议动作 → 让用户 confirm
- 文档一定用中文（面试官快速可读）+ 代码注释中文
- 大文件下载前先测速 → 若 <500 KB/s 需要预警时间

### 你的角色边界

- **能自主做的**：装环境、跑训练、跑评测、写代码/文档、debug 报错
- **需要用户确认的**：花超过 2 小时的任务、下载 >2GB 的文件、动 checkpoint / 训好的权重
- **不能做的**：伪造硬件数据（5060 Ti 数据不能标 4060, 反之亦然）、修改 currbg.tex（用户主简历, 在 `../27提前批和秋招/currbg.tex`，本项目动它需明确许可）

---

## 8 · 附：常用命令速查 + Debug 手册

### 常用命令

```bash
# 每次开新终端
conda activate vlm && cd ~/vlm

# 单次训练（示例 · 参考 §6 P1.1）
python train.py --data data/flickr_1k/flickr_1k_train.jsonl \
                --val-data data/flickr_1k/flickr_1k_val.jsonl \
                --image-root data/flickr_1k/images \
                --steps <N> --batch <N> --lr 1e-3 --dtype bf16 \
                --val-every 50 --out checkpoints/<name>.pt

# 评测
python evaluate.py --ckpt checkpoints/<name>.pt --max-samples 100 --out logs/<name>_eval.json

# 单图推理（debug）
python inference.py --image <path> --ckpt checkpoints/<name>.pt --dtype bf16 --temperature 0.0

# 显存监控
watch -n 1 nvidia-smi

# 训练时看功耗
nvidia-smi --query-gpu=power.draw,memory.used,utilization.gpu --format=csv -l 5
```

### Debug 手册（常见报错）

| 报错关键字 | 原因 | 处理 |
|----------|------|------|
| `CUDA error: no kernel image` | torch 版本不支持 Blackwell | 装 torch >= 2.7 |
| `Repo id must use alphanumeric` | 本地路径给了 transformers 但含反斜杠 | 用 `Path(...).as_posix()` |
| `LocalEntryNotFoundError` | huggingface_hub 尝试联网但失败 | 检查 HF_ENDPOINT / 代理 / 走 curl 手动下 |
| `UnicodeEncodeError: 'gbk'` | Windows 遗留（Ubuntu 不会遇到） | 忽略, 代码有 stdout reconfigure |
| `OSError: We couldn't connect to 'https://hf-mirror.com'` | huggingface_hub v1.23+ xet 协议不兼容 hf-mirror | 换 HF 直连（代理下）或 curl+modelscope |
| CLIPVisionModel `UNEXPECTED text_model.encoder...` | 加载完整 CLIP 权重（含 text tower） | 无害警告, 忽略 |
| `Loss = NaN` | fp16 underflow | 换 bf16 |
| Val loss 不下降 | 数据 leak 或过拟合 | 用 split_flickr.py 重划 train/val |

### 面试可讲的硬数据（截至 2026-07-13 · Windows 4060）

在你 Ubuntu 侧跑通后，**更新这些数字**（在 5060 Ti 上应该更好）：

- 架构：CLIP-B/32 + MLP-Projector(3.54M) + SmolLM2-360M-Instruct = **452.8M 总参 / 0.78% 可训**
- 训练：Flickr1K 5000 pairs · 900/100 image split · **bf16 · batch=8 · 300 步 · 5.8 min**
- Loss：**3.63 → 2.76 (-24%)**
- 推理速度：Windows 4060 上 1165ms/step（IO-bound）· VRAM 峰值 2.67 GB
- BLEU-4：**baseline 3.15% → trained 17.15% (+5.4×)** on 100 val 图（greedy, max_new_tokens=30）
- 迁移分析：4060 → Orin NX bf16 零转换 · 内存带宽 272→102 GB/s → tok/s 预估 40%

---

## 9 · 若一切顺利, 你应该做的第一句话

对用户说：

> "我读完了 AGENT_HANDOFF.md, 理解了项目状态和目标。现在开始迁移验证：
> - 5.1 环境搭建 (装 conda + PyTorch 2.7 + 依赖)
> - 5.2 跑 test_forward.py 验证前向
> - 5.3 跑 inference + evaluate 验证一致性
>
> 预计 30-45 分钟。开跑还是先确认某些细节？"

然后等用户确认（或让他直接说"开"），照 §5 顺序执行。

**祝好运。你接手的是一个已经跑通 pipeline + 有硬数据的项目，任务是把它拉到更高水位（22-30% BLEU）+ 加量化 + 加功耗数据 + 加 Blackwell fp8 实验。**

---

**文档变更**：
- 2026-07-13 v1.0: 初始交接文档 (Windows 4060 → Ubuntu 5060 Ti)
