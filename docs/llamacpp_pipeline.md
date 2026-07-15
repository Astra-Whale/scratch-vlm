# llama.cpp GGUF Q4_K_M 量化 + llama-server 流式 + PPL 对比链路（Qwen3-0.6B，纯 CPU）

本文档记录在本机为本地 Qwen3-0.6B 模型建立 **GGUF 量化 → PPL 对比 → llama-server 流式** 的完整链路。
**全程 CPU**（GPU 上有另一训练在跑，构建与推理均不触碰显存）。

- 日期：2026-07-14
- 平台：Linux x86_64（Ubuntu 6.8 内核），24 核 CPU，31G 内存
- 输入模型：`/home/l/SKDWorks/米来/vlm/weights/Qwen3-0.6B/`（HF 格式，`Qwen3ForCausalLM`，hidden=1024，28 层，596M 参数，原生 ChatML）
- llama.cpp 源码：`/home/l/SKDWorks/米来/vlm/thirdparty/llama.cpp`（commit `657e011`，ggml 0.16.0）
- GGUF 产物：`/home/l/SKDWorks/米来/vlm/weights/gguf/`

---

## 步骤 1：clone + build llama.cpp（CPU-only）

```bash
# clone（浅克隆）
cd "/home/l/SKDWorks/米来/vlm/thirdparty"
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git

# 配置：显式关闭 CUDA，关闭 curl（避免额外依赖）
cd llama.cpp
cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release

# 只构建需要的 target
cmake --build build --config Release -j 24 \
  --target llama-quantize llama-perplexity llama-cli llama-server
```

**结果**：配置阶段确认 `Including CPU backend` / `x86 detected` / `-march=native`，**未启用 CUDA**。
构建 **约 84 秒** 完成（`real 1m24s`），四个二进制全部产出于 `build/bin/`：
`llama-quantize`、`llama-perplexity`、`llama-cli`、`llama-server`。

**踩坑**：
- 构建 `llama-server` 时其内嵌 WebUI 前端资源用本机 node v18 编译失败（依赖要求 node>=20，报 `EBADENGINE`）。llama.cpp 自动回退到从 HuggingFace 下载预编译 `dist.tar.gz`，成功，**不影响 server 功能**。
- 提示未装 `ccache`（仅影响重复编译速度，可忽略）。

---

## 步骤 2：转 f16 GGUF

`convert_hf_to_gguf.py` 需要 python 侧 `gguf` 包；conda env `dl` 未安装，直接用仓库自带的 `gguf-py`（`PYTHONPATH`）。
用 `CUDA_VISIBLE_DEVICES=""` 强制转换过程不占 GPU。

```bash
cd "/home/l/SKDWorks/米来/vlm/thirdparty/llama.cpp"
CUDA_VISIBLE_DEVICES="" PYTHONPATH="gguf-py" conda run -n dl python convert_hf_to_gguf.py \
  "/home/l/SKDWorks/米来/vlm/weights/Qwen3-0.6B/" \
  --outfile "/home/l/SKDWorks/米来/vlm/weights/gguf/qwen3-0.6b-f16.gguf" \
  --outtype f16
```

**结果**：Qwen3 架构被当前 llama.cpp **原生支持**（映射为 `qwen3`），转换成功。
`n_tensors = 311, total_size = 1.5G`，同时把 ChatML 的 chat_template 一并写入 GGUF 元数据。

**版本说明**：转换脚本导入 `torch` / `transformers` / `safetensors` + 仓库内 `gguf-py`。
env `dl` 为 torch 2.11.0+cu128、transformers 5.13.1、python 3.11；虽是 CUDA 版 torch，但转换是纯 CPU numpy 路径，配合 `CUDA_VISIBLE_DEVICES=""` 不初始化显存。

---

## 步骤 3：量化 Q4_K_M

```bash
cd "/home/l/SKDWorks/米来/vlm/thirdparty/llama.cpp"
./build/bin/llama-quantize \
  "/home/l/SKDWorks/米来/vlm/weights/gguf/qwen3-0.6b-f16.gguf" \
  "/home/l/SKDWorks/米来/vlm/weights/gguf/qwen3-0.6b-q4_k_m.gguf" Q4_K_M
```

**结果**：量化 **约 4.3 秒** 完成。日志内部统计：
`model size = 1433.75 MiB (16.00 BPW)` → `quant size = 456.11 MiB (5.09 BPW)`。
（Q4_K_M 混合方案：多数 `attn/ffn` 权重走 q4_K，`attn_v`/`ffn_down` 关键权重升到 q6_K。）

### 文件体积对比

| 文件 | 字节数 | GB (÷1e9) | MiB |
|---|---|---|---|
| `qwen3-0.6b-f16.gguf` | 1,509,347,136 | 1.51 GB | 1439.4 MiB |
| `qwen3-0.6b-q4_k_m.gguf` | 484,219,712 | 0.48 GB | 461.8 MiB |

**压缩比 ≈ 3.12×（体积减少 ≈ 67.9%）。**

---

## 步骤 4：PPL 对比（f16 vs Q4_K_M）

标准语料用 llama.cpp 自带脚本下载的 **wikitext-2-raw**，取 `wiki.test.raw` 前 90000 字节作为小切片
（CPU 上跑得快，约 43 个 512-token chunk），两个模型跑**同一切片、同一参数**。

```bash
# 下载 wikitext-2-raw
cd /tmp && bash "/home/l/SKDWorks/米来/vlm/thirdparty/llama.cpp/scripts/get-wikitext-2.sh"
head -c 90000 /tmp/wikitext-2-raw/wiki.test.raw > /tmp/wiki.slice.raw

# f16 PPL
cd "/home/l/SKDWorks/米来/vlm/thirdparty/llama.cpp"
./build/bin/llama-perplexity -m "/home/l/SKDWorks/米来/vlm/weights/gguf/qwen3-0.6b-f16.gguf" \
  -f /tmp/wiki.slice.raw -c 512 -t 12

# Q4_K_M PPL（同切片同参数）
./build/bin/llama-perplexity -m "/home/l/SKDWorks/米来/vlm/weights/gguf/qwen3-0.6b-q4_k_m.gguf" \
  -f /tmp/wiki.slice.raw -c 512 -t 12
```

### PPL 结果

| 模型 | Perplexity | ± | chunks | ctx |
|---|---|---|---|---|
| f16      | **19.6306** | 0.62130 | 43 | 512 |
| Q4_K_M   | **21.3469** | 0.68574 | 43 | 512 |

**量化引入的 PPL 退化：绝对 +1.716，相对 +8.74%。**

> 说明：PPL 绝对值偏高（约 19–21）是因为这是 **0.6B 小基座模型** 在 wikitext 上、且用了较短的 512 上下文与语料切片；
> 这里关注的是 **f16→Q4_K_M 的相对退化（约 +8.7%）**，对 0.6B 模型来说属正常范围。要更严谨可跑完整 `wiki.test.raw` 并用更大 `-c`。

---

## 步骤 5：llama-server 流式验证（SSE）

```bash
# 起 server：加载 Q4_K_M，-ngl 0 强制 0 层上 GPU（纯 CPU），CUDA_VISIBLE_DEVICES="" 双保险
cd "/home/l/SKDWorks/米来/vlm/thirdparty/llama.cpp"
CUDA_VISIBLE_DEVICES="" ./build/bin/llama-server \
  -m "/home/l/SKDWorks/米来/vlm/weights/gguf/qwen3-0.6b-q4_k_m.gguf" \
  --host 127.0.0.1 --port 8099 -c 2048 -t 12 -ngl 0
# 日志：srv llama_server: listening on http://127.0.0.1:8099
```

### 5a. OpenAI 兼容 `/v1/chat/completions`（stream:true）

```bash
curl -s -N http://127.0.0.1:8099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"Say hello in one short sentence."}],"stream":true,"max_tokens":40,"temperature":0.2}'
```

SSE 流式输出样本（逐 token 到达，Qwen3 思考模式先输出 `reasoning_content`）：

```
data: {"choices":[{"finish_reason":null,"index":0,"delta":{"role":"assistant","content":null}}],...,"object":"chat.completion.chunk"}
data: {"choices":[{"finish_reason":null,"index":0,"delta":{"reasoning_content":"Okay"}}],...,"object":"chat.completion.chunk"}
data: {"choices":[{"finish_reason":null,"index":0,"delta":{"reasoning_content":" the"}}],...}
data: {"choices":[{"finish_reason":null,"index":0,"delta":{"reasoning_content":" user"}}],...}
...
```

### 5b. 原生 `/completion`（stream:true）

```bash
curl -s -N http://127.0.0.1:8099/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","stream":true,"n_predict":12,"temperature":0.1}'
```

SSE 流式输出样本（内容正确：Paris）：

```
data: {"index":0,"content":" Paris","tokens":[12095],"stop":false,"tokens_predicted":1,"tokens_evaluated":5}
data: {"index":0,"content":".","tokens":[13],"stop":false,"tokens_predicted":2,...}
data: {"index":0,"content":" The","tokens":[576],"stop":false,...}
data: {"index":0,"content":" capital","...}
...
```

**结论：SSE 流式 token 输出正常可用**，两个端点（OpenAI 兼容 + 原生）均逐 token 推送。
验证完毕后 `kill` 关闭 server（已确认无残留 `llama-server` 进程，`-ngl 0` 全程未占用显存）。

---

## 附：关键路径速查

| 用途 | 路径 |
|---|---|
| 源码/构建 | `/home/l/SKDWorks/米来/vlm/thirdparty/llama.cpp/build/bin/` |
| f16 GGUF | `/home/l/SKDWorks/米来/vlm/weights/gguf/qwen3-0.6b-f16.gguf` |
| Q4_K_M GGUF | `/home/l/SKDWorks/米来/vlm/weights/gguf/qwen3-0.6b-q4_k_m.gguf` |
| PPL 语料切片 | `/tmp/wiki.slice.raw`（源自 wikitext-2-raw `wiki.test.raw` 前 90KB） |

## 复现要点 / 注意

1. **CPU-only**：`cmake -DGGML_CUDA=OFF`；server 用 `-ngl 0` + `CUDA_VISIBLE_DEVICES=""`；转换脚本也加 `CUDA_VISIBLE_DEVICES=""`。
2. python `gguf` 用仓库自带 `gguf-py`（`PYTHONPATH=gguf-py`），无需额外 pip 安装。
3. WebUI 前端本机 node 版本过低（v18<20）编译失败，但会自动下载预编译资源，server 功能不受影响。
4. PPL/流式部分只涉及**纯 Qwen3 LLM**；多模态 mmproj 集成见下方专节。

---

## mmproj 多模态集成（CLIP + projector → llama.cpp，端到端跑通）

把**自训**的视觉栈（冻结 CLIP-ViT-L/14@336 + 训练好的 2 层 MLP projector）打包成
llama.cpp 的 mmproj GGUF，与 **LoRA 合并后**的 Qwen3 GGUF 组合，在量化运行时里做端到端图文推理。

### 步骤

1. **合并 stage-2 LoRA 进 Qwen3**（`tools_merge_lora.py`）→ 完整 HF 模型 `weights/qwen3_stage2_merged/`
   （用 `ScratchVLM` 保证 `<image>` token resize 与训练一致，再 `merge_and_unload`）。
2. **merged → GGUF f16 → Q4_K_M**：`convert_hf_to_gguf.py` + `llama-quantize`
   → `weights/gguf/qwen3-stage2-merged-q4_k_m.gguf`（**372 MiB**）。
3. **CLIP + projector → mmproj GGUF**（legacy `convert_image_encoder_to_gguf.py`）：
   - 构造 `llava.projector`：本项目 projector 的 `linear1→mm.0`、`linear2→mm.2`（对应 LLaVA `Sequential(Linear, GELU, Linear)` 索引 0/1/2）。
   - `--projector-type mlp`，`--model-dir` 指向完整 CLIP-L/14@336 HF 快照，`PYTHONPATH=gguf-py`。
   - 脚本默认**丢弃 CLIP 最后一层**（到 `blk.22`），正好对应本项目 `VisionEncoder(select_layer=-2)`，**特征层严格一致**。
   → `weights/gguf/mmproj-model-f16.gguf`（**590 MB**，CLIP-L f16 + projector）。
4. **构建**：`cmake --build build --target llama-mtmd-cli`。

### 端到端验证（`llama-mtmd-cli`，`--temp 0`，纯 CPU）

测试图 `COCO_val2014_000000001171.jpg`（实为一辆编号 71 的黑色蒸汽火车头，背景树林山坡）：

| Prompt | 输出 | 判定 |
|---|---|---|
| `Describe this image.` | "a black and white train ... surrounded by a variety of trees ..." | ✅ 图像相关、无幻觉 |
| `Is there a train in the image?` | `Yes` | ✅ 正确 |

**结论**：自训 CLIP+projector+Qwen3(LoRA 合并) 全链路在 llama.cpp **Q4_K_M 量化运行时**里正确工作，
视觉 grounding 穿过量化 LLM 保持准确。这打通了 selfspec 的"CLIP+projector 打包 + 端侧多模态推理"链路。

### 关键坑
- legacy 脚本不在 path 里带 `gguf` 模块 → 需 `PYTHONPATH=<llama.cpp>/gguf-py`。
- **不要**加 `--clip-model-is-vision`：openai CLIP 仓库 config 是嵌套的 `vision_config`/`text_config`，
  该 flag 会把整个 config 误当 vision hparams；走默认路（加载完整 `CLIPModel` 读 `vision_config`）。
- mmproj 的 projector 输出维度必须 = LLM `n_embd`（Qwen3-0.6B = **1024**），本项目 projector 已自动对齐。
