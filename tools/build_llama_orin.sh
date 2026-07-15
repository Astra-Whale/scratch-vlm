#!/usr/bin/env bash
# 在 Jetson Orin (sm_87) 上编译 llama.cpp 的 CUDA 版本。
# 用法: bash tools/build_llama_orin.sh [llama.cpp 目录]
#   不给目录则在 ./thirdparty/llama.cpp 就地编译, 不存在则 clone。
# 前置: JetPack 6.x (CUDA 12.x) + cmake + build-essential。
set -euo pipefail

LLAMA_DIR="${1:-thirdparty/llama.cpp}"

if [ ! -d "$LLAMA_DIR" ]; then
    echo "[clone] $LLAMA_DIR"
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
cd "$LLAMA_DIR"

# Orin = Ampere sm_87; -ngl 99 运行时把所有层 offload 到 GPU (统一内存)
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DGGML_CUDA_F16=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)" \
    --target llama-mtmd-cli llama-server llama-cli llama-quantize llama-perplexity

echo "[done] 二进制在 $LLAMA_DIR/build/bin/"
echo "运行示例:"
echo "  $LLAMA_DIR/build/bin/llama-mtmd-cli -m qwen3-stage2-merged-q4_k_m.gguf \\"
echo "      --mmproj mmproj-model-f16.gguf --image test.jpg -p 'Describe this image.' -ngl 99 --temp 0"
