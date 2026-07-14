# CV 项目条目草稿(可直接粘贴)

> **注**:主简历 `currbg.tex` 在另一仓库、未迁到本机,且按约定改它需你明确许可。
> 这里给出可直接粘贴的条目草稿,数字全部为本项目实测/诚实标注。你确认后自行合入 `currbg.tex`。

---

## 中文版(详细,3-4 条 bullet)

**scratch-vlm · 端侧视觉语言模型全流程 demo**(个人项目,面向端侧 AI 系统)

- 参考 LLaVA v1.5 从零拼装 VLM:冻结 CLIP-ViT-L/14@336 + 冻结 Qwen2.5-0.5B-Instruct,**仅训练 3.94M MLP projector(占总参 0.49%)**;Flickr1K 上 corpus BLEU-4 从未训基线 0.86% 提升至 **20.59%(+24×)**,与经典 Show-Attend-Tell 在 Flickr30k 同量级。
- **混合精度量化**(torchao weight-only,只压 LLM、CLIP/projector 守 bf16):int8 **近乎无损**(BLEU 20.60%)、显存 -14%;int4 -26% 显存;规避非便携 kernel(bitsandbytes / mslk),改用 Ampere 原生 tinygemm 路径守 Jetson 兼容。
- **端侧系统工程**:推理显存仅 **1.6GB**(可部署 Orin Nano 8G);latency 分解(视觉编码 15ms 一次性 + decode 137 tok/s memory-bandwidth-bound);CLIP 前端导出 ONNX(PyTorch→ONNX→TensorRT 部署路径)。
- **跨平台迁移**:同一份代码 4060 Laptop(Windows)→ RTX 5060 Ti(Ubuntu, Blackwell)零改动跑通,VRAM 逐位一致;完成 4060→Jetson Orin NX 迁移可行性分析 + Ampere 兼容性审计(bf16/sdpa/标准算子,无超 Ampere 特性)。

## 英文版(精简,1-2 条 bullet)

**scratch-vlm — End-to-end edge VLM (personal project)**

- Built a LLaVA-v1.5-style VLM from scratch: frozen CLIP-ViT-L/14@336 + frozen Qwen2.5-0.5B-Instruct, training only a **3.94M MLP projector (0.49% of params)**; corpus BLEU-4 0.86%→**20.59% (+24×)** on Flickr1K. Inference in **1.6 GB** VRAM (deployable on Jetson Orin Nano 8G).
- Full edge pipeline: **mixed-precision quantization** (torchao weight-only int8 near-lossless, -14% VRAM; int4 -26%), latency profiling (137 tok/s decode, memory-bandwidth-bound), ONNX export, and zero-code cross-platform migration (4060→5060 Ti) with an Ampere/Orin-NX compatibility audit.

---

## 诚信提醒(合入前自查)
- BLEU 引用务必带"Flickr1K / corpus BLEU-4 / 与 Flickr30k 仅同量级"限定,勿裸报或与 COCO SOTA 比。
- Jetson tok/s 是**预估**,勿写成实测(真机 rerun 后再改)。
- int4 精度回撤(-4.6 点)如提及需一并说明,别只报好的一面。
