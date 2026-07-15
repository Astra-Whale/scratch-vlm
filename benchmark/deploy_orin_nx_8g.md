# 部署到 Jetson Orin NX 8GB

本项目(CLIP-ViT-L/14@336 + MLP projector + Qwen3-0.6B)在 Jetson Orin NX 8GB 上的部署 runbook。
业务代码无需改动;改动仅在环境/构建层。**含真机的数据段为待填模板**(开发机无 Jetson 硬件)。

## 目标设备

| 项 | 值 |
|---|---|
| GPU | 1024-core Ampere, 32 Tensor Cores, **sm_87** |
| CPU | 6-core Arm Cortex-A78AE |
| 内存 | 8GB 128-bit LPDDR5,**统一内存**(CPU/GPU 共享),102 GB/s |
| AI 算力 | 70 TOPS INT8(标准)/ 117(JetPack 6.2 MAXN_SUPER) |
| 功耗档 | 10 / 15 / 20W(+40W Super) |
| bf16 / fp16 / INT8 | ✅ / ✅ / ✅(Ampere 原生);fp8/fp4 ✗ |

sm_87 与开发机 5060 Ti 同属支持 bf16/INT8/sdpa 的世代 → **bf16 权重零转换**(区别于 Xavier NX/Volta 需 bf16→fp16)。

## 内存预算(8GB 统一内存)

JetPack 桌面约占 2–2.5GB(headless 省 ~1GB)。两条路径常驻:

| 路径 | 常驻 | 备注 |
|---|---|---|
| **llama.cpp GGUF(推荐)** | ~1.2–1.5GB | Qwen3 Q4_K_M 0.48GB + mmproj f16 0.59GB + KV(0.6B 极小) |
| PyTorch(ScratchVLM 全精度) | ~2.5–3.5GB | CLIP-L bf16 0.6GB + Qwen3 bf16 1.2GB + 576 视觉 token 激活 |

两条均塞进 8GB;GGUF 路宽裕,建议作为部署主路径。

## 前置

- JetPack 6.2(L4T r36.4,Ubuntu 22.04,CUDA 12.6,cuDNN,TensorRT)。
- `sudo nvpmodel -m 0 && sudo jetson_clocks`(拉满时钟;JetPack 6.2 可启 MAXN_SUPER)。
- 建议 headless(省内存):`sudo systemctl set-default multi-user.target`。

---

## 路径 A(推荐):llama.cpp GGUF + mmproj

产物硬件无关,直接从开发机拷贝:
- `weights/gguf/qwen3-stage2-merged-q4_k_m.gguf`(372 MiB,已合并 stage-2 LoRA)
- `weights/gguf/mmproj-model-f16.gguf`(590 MB,CLIP-L + projector)

### 1. 在 Orin 上编 llama.cpp(CUDA, sm_87)

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DGGML_CUDA_F16=ON
cmake --build build --config Release -j$(nproc)
```
(或用 `jetson-containers` 预构建镜像免编译。)

### 2. 拷贝产物 + 跑

```bash
# 全部层 offload 到 GPU(统一内存, -ngl 99)
./build/bin/llama-mtmd-cli \
    -m qwen3-stage2-merged-q4_k_m.gguf \
    --mmproj mmproj-model-f16.gguf \
    --image test.jpg -p "Describe this image." \
    -ngl 99 --temp 0
```

关键点:`-ngl 99` 把所有层放 GPU;Jetson 统一内存下 GPU offload 用同一物理 RAM,比 CPU 快 2–4×。纯文本对话/流式可起 `llama-server ... -ngl 99`。

---

## 路径 B(可选):PyTorch 全精度

用于在设备上跑完整 ScratchVLM(captioning / POPE 评测)。

### 1. 装 Jetson 版 PyTorch(**不能用 pytorch.org 通用轮子**,否则无 GPU 加速)

```bash
# JetPack 6.2 / CUDA 12.6 / Python 3.10 对应的 NVIDIA aarch64 wheel
# 见 https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/
pip install numpy==1.26.1
pip install torch-2.*-cp310-*aarch64.whl   # NVIDIA Jetson wheel (torch 2.5-2.6)
# cuSPARSELt 0.7.0 为依赖;transformers/peft/pillow/pycocoevalcap 直接 pip
```

### 2. 跑(代码零改动,权重 bf16 直接可用)

```bash
python tests/test_forward.py
python evaluate.py --ckpt checkpoints/projector_stage1_qwen3_best.pt \
    --data data/flickr8k/test.jsonl --image-root data/flickr8k/images --max-samples 100
```

---

## 真机测量模板(待填)

拿到设备后填入,替换本项目 latency/功耗的预估值。

```bash
# 功耗 / 温度 / 利用率(后台采样)
tegrastats --interval 1000 | tee logs/tegrastats_orin.log
```

| 项 | 预估(带宽/架构外推) | 真机实测 |
|---|---|---|
| GGUF Q4 decode tok/s | ~30–50 | _待填_ |
| 视觉编码 + prefill | 几十~100ms 级 | _待填_ |
| 单图 caption 端到端 | 亚秒~1s | _待填_ |
| 常驻内存(GGUF 路) | ~1.2–1.5GB | _待填_ |
| 功耗(15W/20W/Super 档) | — | _待填_ |

预估依据:decode 为 memory-bandwidth-bound,0.48GB / 102 GB/s ≈ 4.7ms/token 理论;对照 5060 Ti(448 GB/s)实测 137 tok/s(bf16)。真机受 JetPack 优化 / kernel 效率 / 统一内存 overhead 影响,可能偏离 ±30%。

## 改动清单小结

| 层 | 是否改动 |
|---|---|
| `model/` · `train*.py` · `evaluate.py` · `inference.py` | **零改动** |
| PyTorch | x86 `torch 2.11+cu128` → Jetson aarch64 wheel(路径 B 才需) |
| llama.cpp | 重编 `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87` |
| GGUF / mmproj / projector 产物 | 硬件无关,直接拷贝 |
| torchao | 端侧不需要(GGUF 已覆盖量化) |
