# 量化方案技术调研 · 决策备忘

**Date:** 2026-07-13
**Owner:** 徐悦
**服务对象:** ROADMAP 主线 **A1 · 混合精度量化**(§6 支线 S6 "给主线排雷")
**范围:** 只调研 + 读代码,**不跑任何模型/训练/GPU 任务**。所有量化实测数字待主线串行执行。

---

## 0 · TL;DR(结论先行)

**先做 `torchao` 权重量化(`int8_weight_only` → `int4_weight_only`),作为主线 A1 的第一落点。**

- 它是**唯一**能:(1) 零新增部署依赖(PyTorch-native,已在 `dl` env)、(2) 就地量化**整条 pipeline**(LLM + projector,甚至 CLIP 的 Linear)、(3) **~5 行接入现有 `evaluate.py` 立即重跑 corpus BLEU**、(4) 天然支撑 ROADMAP A1.3 "混合精度"叙事(LLM 压 int4 / CLIP+projector 守 bf16)的方案。
- Orin 兼容:weight-only 量化在 matmul 时把权重 **upcast 回 bf16/fp16**,走 Ampere sm_87 标准 tensor core,**不碰任何红线**(无 bitsandbytes / 无 fp8-fp4 / 无 Blackwell-only kernel)。
- GGUF Q4_K_M 作为**第二落点(部署路径演示)**,而非首选——它只吃 LLM,本项目的 scratch CLIP+MLP projector **不是 llama.cpp 认识的 mmproj 架构**,无法直接转换;且是独立 runtime,接不进 `evaluate.py`。
- GPTQ / AWQ **不做**:需 calibration data、aarch64 无预编译 wheel(需源码编译)、只吃 LLM、且 AutoGPTQ 与 AutoAWQ 均已停止维护(2025)。相对 torchao 无增量收益,只增成本。

**预计影响(待实测,方向可信):**
- `int8_weight_only`:LLM Linear 权重体积 **≈ ÷2**,corpus BLEU 掉 **≈ 0**(near-lossless,业界经验 <1%)。
- `int4_weight_only`(group=128):LLM Linear 权重体积 **≈ ÷4**,BLEU 可能掉几个点(0.5B 小模型对 int4 更敏感)——用**混合精度**(projector/CLIP 守 bf16)兜底。

---

## 1 · 硬约束回顾(红线,先过闸)

来自 `ROADMAP.md` §1.2 与任务书:

| # | 红线 | 判定 |
|---|------|------|
| 1 | ❌ **bitsandbytes**(aarch64/Jetson 支持差) | 本调研 4 个候选**均非** bnb;但需点名 HF 最常用的 `load_in_4bit`(=bnb NF4)**被排除**——见 §6 |
| 2 | ❌ **fp8 / fp4** | torchao 有 fp8 config,**不用**;TensorRT-LLM fp8、Blackwell fp4 **排除** |
| 3 | ❌ **Blackwell-only kernel** | 只认 Ampere sm_87 + Jetson 可跑;dev 机 5060 Ti(sm_120)仅用于**测体积/精度**(硬件无关、可诚实迁移) |
| 4 | 关注**权重体积压缩 + 精度损失**(硬件无关) | 全部候选按此维度对比,速度/tok/s 归为 Orin 真机实测(标"预估") |

**关键认知:** 体积压缩比 × BLEU 掉多少 = **硬件无关、可诚实迁移**的证据,可在 dev 机(5060 Ti)上测完直接写进交付物;Orin 兼容性是**部署可行性声明**(标注预估/待真机)。二者分开,守 §1.3 诚信底线。

---

## 2 · 四方案对比表

| 维度 | **torchao** ⭐首选 | **GGUF Q4_K_M** (llama.cpp) | **GPTQ** | **AWQ** |
|------|------------------|----------------------------|----------|---------|
| **(a) Orin/aarch64+Ampere 兼容** | ✅ PyTorch-native,sm_87 官方支持(PR #95008);weight-only upcast bf16 走标准 tensor core | ✅ `.gguf` 硬件无关;需源码 build `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87` | ⚠️ 无 aarch64 wheel,需源码编译;Marlin kernel 历史仅 cc 8.0/8.6(Orin 8.7 存疑) | ⚠️ CC≥7.5 架构上 OK,但 autoawq wheel 是 x86,需源码编译 |
| **(b) 能量化整条 pipeline?** | ✅ **全 pipeline**:就地量化 LLM + projector(+ CLIP Linear),同一 `nn.Module` | ❌ **只吃 LLM**;CLIP+projector 需另做 mmproj GGUF,而**本项目 scratch 架构非 llama.cpp 已知 mmproj**,无现成转换器 | ❌ 只吃 LLM(transformers 集成替换 HF 模型 Linear) | ❌ 只吃 LLM |
| **(c) 接入现有 `evaluate.py`?** | ✅ **~5 行**,加载后 `quantize_(model, ...)`,立即重跑 corpus/sentence BLEU | ❌ 独立 C++ runtime;要另写评测 harness | ❌ 需先 calibration 产出量化 ckpt,再改加载路径 | ❌ 同 GPTQ,且推理走 TinyChat/独立 runtime 更顺 |
| **(d) 体积 / 精度** | int8 ÷2(near-lossless);int4 ÷4(小掉,混合精度兜底) | LLM 权重 ÷~3.5-4(Q4_K_M);精度掉小 | int4 ÷4;需好 calibration,精度接近 AWQ | int4 ÷4;activation-aware,精度通常优于 GPTQ |
| **(e) 上手成本** | **最低**:1 个函数调用,无 calibration | 中:build llama.cpp + 转 GGUF + mmproj 卡壳 + Orin "????" bug | 高:calibration data + 源码编译 + 已 deprecated | 高:calibration + 源码编译 + 已 archived |
| **维护状态** | ✅ 活跃(PyTorch 官方) | ✅ 活跃 | ❌ AutoGPTQ deprecated(转 GPTQModel) | ❌ AutoAWQ archived 2025-05(转 llm-compressor) |

---

## 3 · 为什么首选 torchao(展开)

### 3.1 Orin 兼容原理(守红线的核心)
`int8_weight_only` / `int4_weight_only` 是 **weight-only** 量化:权重以 int8/uint4 存储,但 matmul 时 upcast 回 input dtype(`F.linear(input, weight.to(input.dtype))`),**计算仍在 bf16/fp16 tensor core**。因此:
- 不依赖任何 int8/int4 专用 GEMM 硬件路径 → **Ampere sm_87 原生够用**;
- `int4_weight_only` 底层是 tinygemm `torch.ops.aten._weight_int4pack_mm`,是 tensor-core-optimized kernel,Orin 的 Ampere 支持;
- PyTorch 自 PR #95008 起官方支持 sm_87 / Jetson Orin CUDA build。
- **不碰 bitsandbytes / fp8 / fp4 / Blackwell kernel** —— 四条红线全避开。

> ⚠️ **待验证(标注,勿虚标):** torchao 的 tinygemm int4 kernel 的 benchmark 主要在 datacenter GPU(H100/A100)上做,**未见官方 Jetson sm_87 保证**。dev 机 5060 Ti 是 Blackwell(sm_120),int4 tinygemm 需先在本机确认能跑(int8 weight-only 无疑问)。**体积/精度数字与硬件无关,可信迁移;真机 tok/s 待 Orin rerun。**

### 3.2 全 pipeline + 混合精度(直接命中 JD "混合精度量化")
torchao 作用于 `nn.Linear` 模块,`ScratchVLM` 的三段(CLIP / projector / Qwen)都是标准 `nn.Module`,可**选择性**量化:
- **A1.1 baseline:** 只压 LLM(`model.llm`)——权重/计算主体。
- **A1.3 混合精度(叙事王牌):** LLM 压 int4,**projector + CLIP 守 bf16**。因为 projector 是唯一训练过的 3.94M 精华、CLIP 是对齐好的视觉语义,对精度敏感;LLM 冗余大、耐压。一句话讲透"敏感度分析 + 混合精度"。

### 3.3 与 `evaluate.py` 的集成(见 §4)
现有评测已用 `AutoModelForCausalLM` 加载 LLM、手写 corpus/sentence BLEU-4。torchao 就地改模型即可,**评测口径完全不变**,fp16→int8→int4 曲线一键产出。

### 3.4 体积估算(供预期,非实测)
Qwen2.5-0.5B-Instruct ~494M 参数,其中 **embedding(tie_word_embeddings,vocab 151936 × 896 ≈ 136M)不被 weight-only 量化**(只量 Linear)。所以:
- bf16 LLM ≈ 1.0 GB;
- **int8**:Transformer Linear(~358M)÷2,embedding 守 bf16 → LLM ≈ **0.6-0.65 GB**;
- **int4**:Linear ÷4 + group scales,embedding 守 bf16 → LLM ≈ **0.45-0.5 GB**;
- projector 3.94M 可忽略;CLIP 304M 可选 int8(视觉塔对量化较敏感,建议守 bf16)。

> 诚实标注:因 embedding 占比大且不量化,**整模型压缩比小于"权重 ÷4"的直觉值**——报告时按 Linear-only 说明,别虚标整体 ÷4。

---

## 4 · torchao 具体 API / 集成步骤

### 4.1 安装(dev 机 `dl` env,torch 2.11+cu128)
```bash
conda activate dl
pip install torchao          # PyTorch-native,无额外系统依赖
```

### 4.2 最小 API(新版 config 风格,旧 shorthand 仍兼容)
```python
from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig
# 旧别名 int8_weight_only() / int4_weight_only() 亦可

# int8 weight-only(near-lossless,先做这个)
quantize_(model.llm, Int8WeightOnlyConfig())

# int4 weight-only(group_size 默认 128;Qwen 896/4864 均可被 128 整除)
quantize_(model.llm, Int4WeightOnlyConfig(group_size=128))
```
`quantize_` 是**就地(in-place)**替换 `nn.Linear` 的权重与 forward,不改模型结构接口。

### 4.3 接入 `evaluate.py`(建议改动,~5-8 行)
在 `evaluate.py` `main()` 里,模型 `.to(device)`、加载 projector ckpt **之后**、`model.eval()` 附近插入:
```python
# --- 新增 CLI ---
p.add_argument("--quant", choices=["none", "int8", "int4"], default="none",
               help="torchao weight-only 权重量化 (仅量化 LLM, 守 Orin 兼容)")
p.add_argument("--quant-target", choices=["llm", "all"], default="llm",
               help="llm=只压 Qwen(混合精度); all=连 projector/CLIP 一起压")

# --- 加载 projector ckpt 之后 ---
if args.quant != "none":
    from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig
    cfg = Int8WeightOnlyConfig() if args.quant == "int8" else Int4WeightOnlyConfig(group_size=128)
    quantize_(model.llm, cfg)                       # 混合精度: 只压 LLM
    if args.quant_target == "all":
        quantize_(model.projector, cfg)             # 可选: 连 projector 一起(叙事对照)
    print(f"[quant] applied torchao {args.quant} on {args.quant_target}")
```
然后一键出曲线:
```bash
# 三档对照(dev 机, corpus BLEU 口径不变)
python evaluate.py --ckpt checkpoints/projector_L14_qwenInstruct_ft_best.pt --max-samples 100                 # fp16/bf16 baseline
python evaluate.py --ckpt ... --max-samples 100 --quant int8            # int8, LLM only
python evaluate.py --ckpt ... --max-samples 100 --quant int4            # int4, LLM only (混合精度)
python evaluate.py --ckpt ... --max-samples 100 --quant int4 --quant-target all   # 敏感度对照
```
> 显存量测:量化后在同脚本里读 `torch.cuda.max_memory_allocated()` 即可补"推理显存"列。
> **注意 dtype:** int4 tinygemm 要求激活为 bf16/fp16;evaluate.py 已默认 `--dtype bf16`,匹配。

### 4.4 验收产物(对齐 ROADMAP A1 验收)
一张表(fp16 / int8 / int4 / int4-mixed × [体积 · corpus BLEU · sentence BLEU · 推理显存])+ 一条曲线 + 一句结论:
> "int4 weight-only 把 LLM Linear 权重压 ~75%,corpus BLEU 从 20.59% 掉到 X%;混合精度(projector/CLIP 守 bf16)把回撤收窄到 Y%。"

---

## 5 · GGUF Q4_K_M —— 第二落点(部署路径演示,非首选)

**定位:** ROADMAP A1.4 / A3(编译器视角 talk-shop)的**具象部署例子**,不是精度曲线的主力。

**能做:** LLM(Qwen2.5-0.5B)导出 Q4_K_M,在 Orin 上 build llama.cpp(`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87` + `jetson_clocks` + `-ngl 99`)量 tok/s。`.gguf` 硬件无关,4060/Orin/Xavier 通用。

**卡点(必须诚实讲):**
1. **只吃 LLM。** 多模态在 llama.cpp 里需 LLM(`.gguf`)+ **mmproj**(CLIP 投影器 GGUF)两个文件。但 llama.cpp 的 mmproj 转换器只认**已知架构**(LLaVA、Qwen-VL、Qwen3-VL 等);**本项目的 scratch CLIP-L + 自定义 2 层 MLP 不是任何已知 mmproj 架构**,没有现成转换器——要接需自己写 GGUF 图/转换逻辑,成本高。
2. **接不进 `evaluate.py`。** 独立 C++ runtime,重跑 BLEU 要另写 harness,量化前后不同口径。
3. **Orin 已知 bug:** 有报告 Orin NX(JetPack 5.1.2/CUDA 11.4)跑 VL + mmproj 出现 `????????` 乱码输出;需较新 llama.cpp build 规避。

**结论:** 作为"边缘部署路径 + 图编译/算子融合/量化 kernel talk-shop"讲透即可(A3),**不投入到精度曲线主线**。若要真跑,只跑 LLM 部分测 tok/s,mmproj 明确标"需自定义转换,未做"。

---

## 6 · 被排除的方案及原因

| 方案 | 排除原因 |
|------|---------|
| **bitsandbytes**(HF `load_in_4bit` / NF4,QLoRA 同源) | ❌ **红线 1**:aarch64/Jetson 支持差,官方无稳定 Jetson wheel。这是 HF 生态最常被顺手拿来的 int4 路径,**明确不用**。 |
| **fp8 / fp4**(torchao float8、TensorRT-LLM fp8、Blackwell fp4) | ❌ **红线 2**:Ampere sm_87 无原生 fp8;fp4 是 Blackwell-only。torchao 有 fp8 config 也**不启用**。 |
| **GPTQ**(AutoGPTQ / GPTQModel) | 需 calibration data;aarch64 无预编译 wheel,需源码编译 exllama/CUDA kernel;**只吃 LLM**;AutoGPTQ 已 deprecated;Marlin kernel 历史仅 cc 8.0/8.6(Orin 8.7 兼容存疑,vLLM 容器实测可选 gptq_marlin 但依赖重)。相对 torchao **无增量精度收益,纯增成本**。 |
| **AWQ**(AutoAWQ / MIT llm-awq TinyChat) | 需 calibration;AutoAWQ 已 **archived(2025-05)**(转 llm-compressor);autoawq wheel 是 x86,aarch64 需源码编译;**只吃 LLM**;MIT TinyChat 虽有一流 Orin 支持(38 tok/s)但是**独立 runtime**,接不进 `evaluate.py`。**留作 S5 竞品调研谈资**(AWQ 是 edge VLM SOTA 部署路径),本项目精度曲线不走它。 |
| **TensorRT-LLM INT4 AWQ** | NVIDIA 对 Orin 的推荐生产路径,但量化 export 需在 x86/Thor 上做(Orin 显存不够 export);且是重型独立部署栈。**留作 talk-shop / 未来生产化**,非当前主线。 |

---

## 7 · 执行建议(给主线 A1)

1. **P2-a(先):** `pip install torchao` → 改 `evaluate.py` 加 `--quant`(§4.3)→ 在 dev 机跑 fp16/int8/int4/int4-mixed 四档,出**体积 × corpus BLEU × 显存**表 + 曲线。**~0.5 天,零红线风险。**
2. **P2-b(混合精度叙事):** `--quant int4 --quant-target all` vs `--quant-target llm` 对照,证明"projector/CLIP 守 bf16"的价值 → 直接对上 JD "混合精度量化 + 敏感度分析"。
3. **P3(可选,部署路径):** LLM 导 GGUF Q4_K_M,Orin build llama.cpp 量 tok/s(mmproj 标"需自定义转换,未做");作为 A3 编译器视角 talk-shop 素材。
4. **诚信:** dev 机数字标"体积/精度(硬件无关)";任何 Orin tok/s 标"预估/待真机";corpus + sentence 双报。

---

## 8 · 实测结果(2026-07-13, 5060 Ti, 100 val, torchao 0.17)

按 §7 执行完毕。`evaluate.py --quant {none,int8,int4}`,SOTA ckpt `projector_L14_qwenInstruct_ft_best.pt`。
**关键配置:排除 tied lm_head(见下方发现),weight-only 只压 Qwen 的 transformer Linear:**

| 量化 | corpus BLEU-4 | sentence BLEU-4 | 权重常驻显存(整模型) | vs bf16 |
|------|--------------|-----------------|---------------------|---------|
| none (bf16) | **20.59%** | 22.18% | 1536 MB | — |
| **int8** weight-only | 20.04% | 21.81% | 1191 MB | **-22% 显存, 近乎无损(±0.5 贪心噪声)** |
| **int4** weight-only (g=128) | 18.33% | 20.10% | 1057 MB | -31% 显存, **corpus BLEU -2.3 点** |

**结论:**
- **int8 近乎无损** + 显存 -22%,直接采用。
- **int4** 用 2.3 BLEU 点换 31% 显存;内存极限场景可选。
- 这是**混合精度**:weight-only 只压 Qwen transformer Linear,**CLIP-L + projector + lm_head 全程守 bf16** → 直接对上 JD "混合精度量化 + 敏感度分析"。

**核心发现:tied embedding × 量化的坑(把结果做得更好的关键)**

排除 lm_head 前后对比(int8,LLM 单独测):
- 量化**全部** Linear(含 lm_head):950→736MB,降 **215MB**;
- **排除** lm_head:降 **346MB**(1.6×)。

原因:Qwen `tie_word_embeddings=True`,lm_head 与 embedding **共享同一份 bf16 权重(136M/272MB)**。torchao 量化 lm_head 会**打破共享**——凭空新增一份 int8 副本(+136MB),抵消近一半收益。且 lm_head 是映射到 151936 词表的**精度敏感输出层**,量化它对 int4 伤害大(corpus BLEU 15.97 → 排除后 **18.33**,+2.4 点)。**排除 tied/敏感的 lm_head 是一举两得:省更多显存 + 保更高精度**,正是混合精度敏感度分析的实证。

**两个诚实标注:**
1. **整模型总显存降幅仍受 CLIP-L 限制**:int4 -31% 已不错,但 CLIP-ViT-L/14(~608MB bf16)未量化,是剩余最大的 bf16 块。**要更深的端侧压缩,下一步是量化 CLIP**(视觉塔对量化更敏感,需谨慎)。量化 LLM 只是第一步。
2. **int4 packing 踩坑(Orin 红线实战)**:torchao 0.17 默认 int4 packing 需 `mslk`(仅 0.0.0 占位版,非便携)→ 撞 Orin 红线。改用 `int4_packing_format="tile_packed_to_4d"`(tinygemm,Ampere sm_80+ 原生、Orin 兼容)。

**未测(诚实留白)**:量化后 decode 延迟未量化——weight-only 在 matmul 时 upcast 回 bf16,主要省显存而非必然提速(kernel 相关);Orin 真机 tok/s 待 rerun。

---

## 附:参考来源

- torchao: [pytorch/ao](https://github.com/pytorch/ao) · [Int4WeightOnlyConfig 文档](https://docs.pytorch.org/ao/stable/generated/torchao.quantization.Int4WeightOnlyConfig.html) · [Quantized Inference](https://docs.pytorch.org/ao/stable/workflows/inference.html) · [HF transformers torchao](https://huggingface.co/docs/transformers/en/quantization/torchao) · [Accelerating LLM Inference (GemLite/tinygemm)](https://pytorch.org/blog/accelerating-llm-inference/)
- Jetson sm_87: [pytorch PR #95008](https://github.com/pytorch/pytorch/pull/95008) · [TensorRT Support Matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)
- llama.cpp / Jetson / mmproj: [Installing llama.cpp Orin NX](https://forums.developer.nvidia.com/t/installing-llama-cpp/354328) · [LLM on Jetson Orin (llama.cpp/Ollama)](https://proventusnova.com/blog/llm-inference-jetson-orin-llamacpp-ollama/) · [Orin NX VL "????" bug #17023](https://github.com/ggml-org/llama.cpp/issues/17023)
- GPTQ: [GPTQModel](https://github.com/modelcloud/gptqmodel) · [auto-gptq (deprecated)](https://pypi.org/project/auto-gptq/) · [transformers issue #36139 (GPTQ on Orin Nano)](https://github.com/huggingface/transformers/issues/36139) · [jetson-containers #678 (Qwen2.5-GPTQ)](https://github.com/dusty-nv/jetson-containers/issues/678)
- AWQ: [mit-han-lab/llm-awq (TinyChat, Orin)](https://github.com/mit-han-lab/llm-awq) · [AutoAWQ (archived)](https://github.com/casper-hansen/AutoAWQ) · [TensorRT Edge-LLM on Jetson (INT4 AWQ)](https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/) · [Qwen AWQ 文档](https://qwen.readthedocs.io/en/latest/quantization/awq.html)
</content>
</invoke>
