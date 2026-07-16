#!/usr/bin/env bash
# 从 stage-2 LoRA 一键构建端侧 GGUF 全链路:
#   1) 合并 LoRA 进 Qwen3         -> weights/qwen3_stage2_merged/
#   2) 合并模型 -> f16 GGUF        -> weights/gguf/qwen3-stage2-merged-f16.gguf
#   3) 量化 Q4_K_M                -> weights/gguf/qwen3-stage2-merged-q4_k_m.gguf
#   4) CLIP + projector -> mmproj  -> weights/gguf/mmproj-model-f16.gguf
#
# 前置: conda env dl;llama.cpp 已编(见 tools/build_llama_orin.sh 或 docs/llamacpp_pipeline.md)。
# 用法: bash tools/build_gguf.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
LC="$ROOT/thirdparty/llama.cpp"
GGUF="$ROOT/weights/gguf"; mkdir -p "$GGUF"
PROJ_CKPT="checkpoints/vlm_stage2_mix2/projector.pt"

echo "[1/4] 合并 LoRA -> HF 模型"
conda run --no-capture-output -n dl python tools/merge_lora.py \
    --lora checkpoints/vlm_stage2_mix2/lora_adapter --out weights/qwen3_stage2_merged

echo "[2/4] HF -> f16 GGUF"
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$LC/gguf-py" conda run --no-capture-output -n dl \
    python "$LC/convert_hf_to_gguf.py" weights/qwen3_stage2_merged \
    --outfile "$GGUF/qwen3-stage2-merged-f16.gguf" --outtype f16

echo "[3/4] 量化 Q4_K_M"
"$LC/build/bin/llama-quantize" \
    "$GGUF/qwen3-stage2-merged-f16.gguf" "$GGUF/qwen3-stage2-merged-q4_k_m.gguf" Q4_K_M

echo "[4/4] CLIP + projector -> mmproj GGUF"
# 4a. 把 projector 权重改名成 llava mm.0/mm.2, 落到 CLIP HF 快照目录
SNAP=$(conda run -n dl python - <<'PY'
import torch, json, glob, os
from huggingface_hub import snapshot_download
snap = snapshot_download("openai/clip-vit-large-patch14-336")
ck = torch.load("checkpoints/vlm_stage2_mix2/projector.pt", map_location="cpu", weights_only=False)
sd = ck["projector_state_dict"]
proj = {"mm.0.weight": sd["linear1.weight"].float(), "mm.0.bias": sd["linear1.bias"].float(),
        "mm.2.weight": sd["linear2.weight"].float(), "mm.2.bias": sd["linear2.bias"].float()}
torch.save(proj, os.path.join(snap, "llava.projector"))
print(snap)
PY
)
# 4b. legacy 转换器(注意: 不加 --clip-model-is-vision;需 PYTHONPATH=gguf-py)
PYTHONPATH="$LC/gguf-py" conda run --no-capture-output -n dl python \
    "$LC/tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py" \
    --model-dir "$SNAP" --llava-projector "$SNAP/llava.projector" \
    --projector-type mlp --output-dir "$GGUF"

echo "[done] GGUF 产物:"; ls -la "$GGUF"/*.gguf
