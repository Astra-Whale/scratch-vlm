# 迁移到 Ubuntu + RTX 5060 Ti 16G · 执行手册

**目标**：把 `D:\SKDWorks\米来\vlm\` 从 Windows/RTX 4060 迁到 Ubuntu/RTX 5060 Ti 16G。
**估计总时间**：1.5-2.5 小时（含 PyTorch 装、模型/数据传输、验证）。

---

## 步骤 0 · Ubuntu 侧前置检查

打开 Ubuntu terminal，跑：

```bash
# GPU + driver
nvidia-smi

# CUDA runtime (可能没有，可选)
nvcc --version 2>/dev/null || echo "no nvcc, will use pytorch-bundled CUDA"

# Python + conda
python3 --version
which conda || echo "no conda, will install"

# 硬盘剩余
df -h ~
```

**预期输出关键点**：
- `nvidia-smi` 显示 "NVIDIA GeForce RTX 5060 Ti" + Driver >= **570** (Blackwell 需要新驱动)
- CUDA Version 显示 **>= 12.6**（driver 声明的支持上限）
- `python3 --version` >= 3.10（或没关系，conda 会自己带）
- 硬盘 >= 5 GB 空闲

如果 driver 太老，先装：
```bash
sudo ubuntu-drivers install --gpgpu nvidia:570   # 或 nvidia:575 更新版
sudo reboot
```

---

## 步骤 1 · 装 Miniconda（若没）

```bash
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc  # 或重启 terminal
conda --version   # 验证
```

---

## 步骤 2 · 建新 conda env (Blackwell 需 PyTorch 2.7+)

**关键**：**PyTorch 2.6 不支持 Blackwell sm_120**，必须 2.7 或更新（发布于 2025 Q1）。

```bash
conda create -n vlm python=3.10 -y
conda activate vlm

# 装 PyTorch 2.7+ with CUDA 12.6
# 官方 index (若代理生效应正常):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 或如果慢, 用清华镜像 (只对 pip 主包起作用, CUDA whl 得原站):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 装项目依赖
pip install transformers accelerate sentencepiece pillow modelscope
```

---

## 步骤 3 · 验证 CUDA + Blackwell 识别

```bash
python -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0))
print('compute cap:', torch.cuda.get_device_capability(0))
print('VRAM (GB):', torch.cuda.get_device_properties(0).total_memory / 1024**3)
print('supports bf16:', torch.cuda.is_bf16_supported())
"
```

**期望输出**：
```
torch: 2.7.x+cu126
cuda available: True
device: NVIDIA GeForce RTX 5060 Ti
compute cap: (12, 0)   ← sm_120 = Blackwell
VRAM (GB): 15.9        ← 或 16.0
supports bf16: True
```

**若 `torch.cuda.is_available() = False` 或报 "no CUDA GPUs"**：可能是 driver/torch 版本不匹配。查看 `nvidia-smi` 显示的 CUDA Version，再装对应 torch 版本。

---

## 步骤 4 · 传输项目到 Ubuntu

三种方式任选：

### 方式 A · U 盘（最简单，若两机不联网）
- Windows 侧：把 `D:\SKDWorks\米来\vlm\` **整个文件夹** 拷到 U 盘（~830 MB 含 605MB CLIP + 200MB Flickr + 15MB checkpoints + 代码）
- 排除 `data/flickr_1k/images_flickr_1k_test.zip`（134MB 已解压，不用）
- Ubuntu 侧：`cp -r /media/USB/vlm ~/vlm`

### 方式 B · scp/rsync（若两机联网 SSH）
在 Ubuntu 侧：
```bash
# 假设 Windows 侧 Cygwin/WSL 能 ssh 或用 Windows OpenSSH
# 或用 SMB 挂载
mkdir -p ~/vlm
scp -r <windows_user>@<windows_ip>:/d/SKDWorks/米来/vlm/{model,data,tests,checkpoints,models,logs,benchmark} ~/vlm/
scp <windows_user>@<windows_ip>:/d/SKDWorks/米来/vlm/*.py ~/vlm/
scp <windows_user>@<windows_ip>:/d/SKDWorks/米来/vlm/*.md ~/vlm/
scp <windows_user>@<windows_ip>:/d/SKDWorks/米来/vlm/{requirements.txt,.gitignore} ~/vlm/
```

### 方式 C · 只拷代码 + 到 Ubuntu 侧重新下模型/数据（省 U 盘容量）
- Windows 侧只拷 **代码 + checkpoint** (~15 MB)：`model/`、`data/dataset.py`、`data/split_flickr.py`、`data/prepare_flickr.py`、`data/toy_dataset.py`、`tests/`、`train.py`、`inference.py`、`evaluate.py`、`README.md`、`benchmark/`、`checkpoints/`
- Ubuntu 侧重新下：
  - CLIP-B/32：`curl -L -o models/models/openai-mirror--clip-vit-base-patch32/snapshots/master/pytorch_model.bin ...`（原始 modelscope URL）
  - SmolLM2-360M-Instruct：`huggingface-cli download HuggingFaceTB/SmolLM2-360M-Instruct`（走代理，HF API 直连应能过）
  - Flickr1K：`curl -L -o data/flickr_1k/images_flickr_1k_test.zip https://hf-mirror.com/datasets/nlphuji/flickr_1k_test_image_text_retrieval/resolve/main/images_flickr_1k_test.zip` + 跑 `python data/prepare_flickr.py`

**推荐方式 A**（简单粗暴）或 **C**（若网速能忍）。

---

## 步骤 5 · 验证跑通

```bash
cd ~/vlm

# 5.1 · 前向验证
python tests/test_forward.py
```

**期望**：
- `[env] device=cuda, dtype=torch.float16`
- 前向 loss 有值
- **VRAM 峰值 <1.5GB**（比 Windows 4060 更省, 因为 desktop 版没热降频）

```bash
# 5.2 · 现有 checkpoint 推理验证
python inference.py --image data/toy_images/000_orange_square.png --dtype bf16 --temperature 0.0
```

**期望**：输出 `A green triangle.` 或类似（迁移一致性验证）

```bash
# 5.3 · 评测跑 20 张 val
python evaluate.py --ckpt checkpoints/projector_flickr1k.pt --max-samples 20 --show-samples 3
```

**期望**：BLEU-4 在 0.14 附近（跟 4060 上的 100 张 avg 0.17 差不多; 20 张会波动）

---

## 步骤 6 · 若上面全通 → 5060 Ti 升级实验清单

三件套升级（按性价比排序）：

### 6.1 · Flickr1K 5 epoch 完整训练（30 分钟, 无下载）
```bash
python train.py \
  --data data/flickr_1k/flickr_1k_train.jsonl \
  --val-data data/flickr_1k/flickr_1k_val.jsonl \
  --image-root data/flickr_1k/images \
  --steps 1500 --batch 16 --lr 1e-3 --dtype bf16 \
  --val-every 50 --val-batches 8 \
  --out checkpoints/projector_flickr1k_5ep.pt
```

**注意**：`--batch 16` 是 5060 Ti 16G 的甜点（4060 上跑不了）。VRAM 应该占 4-5 GB。

### 6.2 · CLIP-L/14 @ 336px 升级（1-1.5 小时，含下载）
- 下载 CLIP-L/14: `huggingface-cli download openai/clip-vit-large-patch14-336`（约 1.7 GB, 通过代理速度 400 KB/s-3 MB/s）
- 或走 modelscope: `AI-ModelScope/clip-vit-large-patch14` (前面验证过存在)
- 重训：`python train.py --vision openai/clip-vit-large-patch14-336 --steps 1500 --batch 8 ...`

### 6.3 · LLM 换 Qwen2.5-0.5B（30 分钟）
- 下载 Qwen2.5-0.5B：约 1 GB
- 重训：`python train.py --llm Qwen/Qwen2.5-0.5B --steps 1500 --batch 8 ...`
- **注意**：Qwen tokenizer 也有 `<|im_start|>` ChatML，但 verify 一下 `dataset.py` 的 special tokens 兼容

### 6.4 · Blackwell fp8 实验（未来做，需装 transformer-engine）
```bash
pip install transformer-engine[pytorch]  # NVIDIA 官方 fp8 支持库
```
- 面试 talk shop 加分项

---

## 常见坑

### PyTorch 报 "GPU not supported"
- 原因：torch 2.6 或更老不支持 Blackwell (sm_120)
- 解决：`pip install torch>=2.7 torchvision --index-url https://download.pytorch.org/whl/cu126`

### `nvidia-smi` 显示 CUDA Version 但 `torch.cuda.is_available() = False`
- 原因：torch whl 版本与驱动不匹配
- 解决：查 `nvidia-smi` 里 "CUDA Version: 12.x"，装对应 cu12x 版 torch

### 中文路径引发 UTF-8 问题
- Ubuntu 一般默认 UTF-8, 不像 Windows GBK
- 若报 `UnicodeEncodeError`, 加 `export LANG=en_US.UTF-8` 或代码里 `sys.stdout.reconfigure(encoding='utf-8')` 已存在

### 磁盘挂载
- 若 Windows 分区可读, Ubuntu 上路径可能是 `/mnt/d/SKDWorks/...`（WSL 风格）或 `/media/<user>/<drive>/...`
- 建议独立拷贝到 `~/vlm/` 不依赖 Windows 分区

---

## 迁移完的对比测试（面试可讲）

在 Ubuntu 5060 Ti 上跑同样任务，与 4060 Laptop 对比：

| 测试 | 4060 Laptop (Windows) | **5060 Ti (Ubuntu)** | 提升 |
|-----|----------------------|----------------------|-----|
| test_forward VRAM | 1.10 GB | ? | ? |
| Flickr 300 步训练 wall time | 5.8 min | ? | ? (预估 3-4 min) |
| 100 张 val evaluation | 87s | ? | ? |
| batch=8 vs batch=16 | 无法上 16 | 应可以 | 更大 batch |
| bf16 loss 稳定性 | 有一定波动 (2.4-3.2) | ? | ? |

**跑完后填这张表**，成为"跨平台迁移验证" 的硬数据，面试极值钱。

---

## 下一步（若一切顺利）

1. **6.1 (5 epoch) 是最快见效**，先跑
2. **6.2 (CLIP-L/14)** 是次优, 下载完再跑
3. **6.4 (fp8 实验)** 长线加分, 有空再玩
4. **回到主对话**（Windows Claude Code 或 Ubuntu 上新开 Claude Code）汇报数据
