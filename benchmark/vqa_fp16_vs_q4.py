"""fp16 vs Q4_K_M 的 VQA 定性对照(端侧量化精度代价的直观对照)。

对同一批图、同一组 prompt,分别用 f16 与 Q4_K_M 的合并模型(+ mmproj)经
llama-mtmd-cli 生成,并排记录输出,产出 logs/vqa_fp16_vs_q4.json。
与 PPL(数值)互补:PPL 给量化的整体退化量,本对照给逐样本的可读差异。

前置:llama.cpp 已编 llama-mtmd-cli;GGUF 见 weights/gguf/。
用法:
  python benchmark/vqa_fp16_vs_q4.py --n 8
"""
import os
import json
import argparse
import subprocess
from pathlib import Path

_ROOT = Path(__file__).parent.parent
CLI = _ROOT / "thirdparty/llama.cpp/build/bin/llama-mtmd-cli"
MMPROJ = _ROOT / "weights/gguf/mmproj-model-f16.gguf"
F16 = _ROOT / "weights/gguf/qwen3-stage2-merged-f16.gguf"
Q4 = _ROOT / "weights/gguf/qwen3-stage2-merged-q4_k_m.gguf"
PROMPTS = ["Describe this image.", "What is the main subject of this image?"]


def run(model, image, prompt, ngl):
    cmd = [str(CLI), "-m", str(model), "--mmproj", str(MMPROJ),
           "--image", str(image), "-p", prompt,
           "-n", "40", "--temp", "0", "-ngl", str(ngl)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    # llama-mtmd-cli 把生成写到 stdout;取非空行的末段作为回答
    lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-root", default="data/coco/val2014")
    p.add_argument("--n", type=int, default=8, help="对照图片数")
    p.add_argument("--ngl", type=int, default=0, help="GPU offload 层数(0=CPU)")
    p.add_argument("--out", default="logs/vqa_fp16_vs_q4.json")
    args = p.parse_args()

    imgs = sorted(Path(args.image_root).glob("*.jpg"))[:args.n]
    results = []
    for i, img in enumerate(imgs):
        for prompt in PROMPTS:
            fp16_out = run(F16, img, prompt, args.ngl)
            q4_out = run(Q4, img, prompt, args.ngl)
            results.append({"image": img.name, "prompt": prompt,
                            "fp16": fp16_out, "q4_k_m": q4_out,
                            "identical": fp16_out == q4_out})
            print(f"[{i+1}/{len(imgs)}] {img.name} | {prompt}")
            print(f"  fp16: {fp16_out}")
            print(f"  q4  : {q4_out}")

    n_id = sum(r["identical"] for r in results)
    summary = {"n_pairs": len(results), "identical": n_id,
               "identical_ratio": round(n_id / max(1, len(results)), 3),
               "models": {"fp16": F16.name, "q4_k_m": Q4.name, "mmproj": MMPROJ.name},
               "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"\n[out] {len(results)} 对 ({n_id} 完全一致) -> {args.out}")


if __name__ == "__main__":
    main()
