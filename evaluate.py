"""
VLM 评测脚本 · BLEU-4 + baseline 对比

对 val set (default: data/flickr8k/test.jsonl) 每张图:
    1. 生成 caption (greedy, max 30 tokens)
    2. 与该图的 5 条 GT caption 算 max BLEU-4
    3. 汇总 corpus 平均 BLEU-4

支持 baseline 模式 (--baseline):
    不加载 projector 权重, 用随机初始化 projector 跑一遍,
    对比 "训后 vs 未训" 效果差异, 证明训练确实有价值。

BLEU-4 手工实现 (免第三方依赖):
    - Modified n-gram precision (1..4)
    - Brevity penalty
    - Chen & Cherry method-1 smoothing (对 zero-count n-gram)

用法:
    # 评测训好的 projector
    python evaluate.py --ckpt checkpoints/projector_stage1_qwen3_best.pt

    # 未训基线对比
    python evaluate.py --baseline

    # 前 20 张图快速试
    python evaluate.py --ckpt checkpoints/projector_stage1_qwen3_best.pt --max-samples 20
"""
import os
import sys
import json
import math
import re
import argparse
import time
from pathlib import Path
from collections import Counter, defaultdict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from PIL import Image

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from model.vlm import ScratchVLM, IMAGE_TOKEN


# ==========================================================
# BLEU-4 (手写, 免依赖)
# ==========================================================
def tokenize(text: str) -> list:
    """小写 + 按非字母数字切分。BLEU 标准做法之一。"""
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(tokens: list, n: int) -> list:
    if len(tokens) < n:
        return []
    return list(zip(*[tokens[i:] for i in range(n)]))


def sentence_bleu4(candidate_tokens: list, refs_tokens: list) -> float:
    """
    Multi-reference sentence BLEU-4 with Chen & Cherry method-1 smoothing.

    Args:
        candidate_tokens: 待评句 token list
        refs_tokens: 若干条 reference token lists (至少 1 条)

    Returns:
        BLEU-4 in [0, 1]
    """
    c_len = len(candidate_tokens)
    if c_len == 0 or not refs_tokens:
        return 0.0

    # Effective ref length = 与 candidate 最接近的 reference 长度
    r_len = min(refs_tokens, key=lambda r: (abs(len(r) - c_len), len(r)))
    r_len = len(r_len)

    # Brevity penalty
    if c_len >= r_len:
        bp = 1.0
    else:
        bp = math.exp(1 - r_len / c_len)

    # Modified n-gram precision, n=1..4
    precisions = []
    for n in range(1, 5):
        c_ngrams = Counter(ngrams(candidate_tokens, n))
        if not c_ngrams:
            precisions.append((0, 0))
            continue

        # Union of max ref counts per n-gram
        max_ref = Counter()
        for r in refs_tokens:
            r_ngrams = Counter(ngrams(r, n))
            for k, v in r_ngrams.items():
                max_ref[k] = max(max_ref[k], v)

        clipped = sum(min(v, max_ref[k]) for k, v in c_ngrams.items())
        total = sum(c_ngrams.values())
        precisions.append((clipped, total))

    # Chen & Cherry method-1 smoothing: 对 zero-count 加 1/(2^k)
    smoothed = []
    invcnt = 1.0
    for i, (clipped, total) in enumerate(precisions):
        if total == 0:
            smoothed.append(0.0)
        elif clipped > 0:
            smoothed.append(clipped / total)
        else:
            invcnt /= 2.0
            smoothed.append(invcnt / total)

    # Geometric mean
    if all(p > 0 for p in smoothed):
        log_avg = sum(math.log(p) for p in smoothed) / 4.0
        return bp * math.exp(log_avg)
    else:
        return 0.0


def corpus_bleu4(results: list):
    """标准 corpus-level BLEU-4 (无平滑, brevity penalty on corpus 总长)。

    这是 captioning 论文 (Show-and-Tell 等) 通用的报告方式, 可跨论文对标。
    与 sentence_bleu4 的 sentence-level 平均 + 平滑不同: 后者对短句更宽容、
    数值偏高, 不能直接和论文比。两者都报以保持透明。

    Returns: (bleu4, [p1, p2, p3, p4])
    """
    clip = [0, 0, 0, 0]
    total = [0, 0, 0, 0]
    c_len = 0
    r_len = 0
    for r in results:
        c = tokenize(r["gen"])
        refs = [tokenize(x) for x in r["refs"]]
        if not refs:
            continue
        if not c:
            # 空生成: 仍计入最接近的 ref 长度以正确施加 brevity penalty
            r_len += min(len(x) for x in refs)
            continue
        c_len += len(c)
        # effective reference length: 长度最接近 candidate 的 ref
        best_ref = min(refs, key=lambda x: (abs(len(x) - len(c)), len(x)))
        r_len += len(best_ref)
        for n in range(1, 5):
            c_ngrams = Counter(ngrams(c, n))
            if not c_ngrams:
                continue
            max_ref = Counter()
            for ref in refs:
                for k, v in Counter(ngrams(ref, n)).items():
                    max_ref[k] = max(max_ref[k], v)
            clip[n - 1] += sum(min(v, max_ref[k]) for k, v in c_ngrams.items())
            total[n - 1] += sum(c_ngrams.values())

    precisions = [(clip[i] / total[i]) if total[i] > 0 else 0.0 for i in range(4)]
    if c_len == 0 or not all(p > 0 for p in precisions):
        return 0.0, precisions
    bp = 1.0 if c_len >= r_len else math.exp(1 - r_len / c_len)
    bleu = bp * math.exp(sum(math.log(p) for p in precisions) / 4.0)
    return bleu, precisions


# ==========================================================
# 主流程
# ==========================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/projector_stage1_qwen3_best.pt",
                   help="projector checkpoint")
    p.add_argument("--data", type=str, default="data/flickr8k/test.jsonl",
                   help="val/test JSONL")
    p.add_argument("--image-root", type=str, default="data/flickr8k/images",
                   help="图像根目录")
    p.add_argument("--max-samples", type=int, default=1000,
                   help="评测多少张 unique 图")
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--baseline", action="store_true",
                   help="不加载 projector (未训 baseline 对比)")
    p.add_argument("--show-samples", type=int, default=5,
                   help="打印多少条 caption sample")
    p.add_argument("--out", type=str, default=None,
                   help="结果保存 JSON 路径 (可选)")
    p.add_argument("--vision", type=str, default=None,
                   help="覆盖 vision 模型路径 (默认从 ckpt 读; 跨平台迁移时用来覆盖 ckpt 里的旧绝对路径)")
    p.add_argument("--llm", type=str, default=None,
                   help="覆盖 llm (默认从 ckpt 读)")
    p.add_argument("--quant", type=str, default="none", choices=["none", "int8", "int4"],
                   help="对 LLM 做 torchao weight-only 量化 (Orin 兼容; 禁 bitsandbytes)。"
                        "int8 near-lossless; int4 在 torchao 0.17 需非便携 kernel(mslk), 默认排除")
    return p.parse_args()


def load_val(data_path: str) -> dict:
    """读 JSONL, 按 image 聚合成 {image: [caption1, caption2, ...]}。"""
    by_image = defaultdict(list)
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_image[r["image"]].append(r["caption"])
    return dict(by_image)


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # 加载 ckpt (若非 baseline)
    ckpt = None
    if not args.baseline:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            print(f"[error] ckpt 不存在: {ckpt_path}")
            print(f"        若想跑未训 baseline, 加 --baseline")
            sys.exit(1)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        vision_name = ckpt.get("vision_name")
        llm_name = ckpt.get("llm_name")
        print(f"[ckpt] {ckpt_path}: trained_steps={ckpt.get('trained_steps')}, "
              f"final_loss={ckpt.get('final_loss')}")
    else:
        vision_name = "openai/clip-vit-large-patch14-336"
        llm_name = "Qwen/Qwen3-0.6B"
        print("[baseline] 未加载 projector, 用随机初始化跑基线")

    # ---- CLI 覆盖 (跨平台迁移: ckpt 里可能存的是旧机器的绝对路径) ----
    if args.vision:
        vision_name = args.vision
    if args.llm:
        llm_name = args.llm

    # ---- 本地路径 fallback: ckpt 存的 *绝对路径* 若在本机不存在, 回退到本地默认 CLIP ----
    # 只针对绝对路径 / Windows 盘符路径 (真正的迁移问题, 如 'D:/...'); HF repo_id
    # (如 'openai/clip-vit-large-patch14-336') 也含 '/', 绝不能误判成本地路径。
    _is_local_path = vision_name and (
        vision_name[0] in "/\\"
        or "\\" in vision_name
        or (len(vision_name) > 1 and vision_name[1] == ":")
    )
    if _is_local_path and not Path(vision_name).exists():
        print(f"[warn] ckpt 记录的 vision 路径不存在 ({vision_name}), 回退到 openai/clip-vit-large-patch14-336")
        vision_name = "openai/clip-vit-large-patch14-336"

    # 加载 model
    print(f"[model] vision={vision_name}")
    print(f"[model] llm={llm_name}")
    t0 = time.time()
    model = ScratchVLM(
        vision_model_name=vision_name,
        llm_model_name=llm_name,
        dtype=dtype,
        device=device,
    ).to(device)
    model.eval()
    if ckpt is not None:
        model.projector.load_state_dict(ckpt["projector_state_dict"])
        print(f"[ckpt] projector 权重加载完成")
    print(f"[model] setup 耗时 {time.time() - t0:.1f}s")

    # 可选: torchao weight-only 量化 LLM (Orin 兼容路径, 禁 bitsandbytes)
    if args.quant != "none":
        import torch.nn as nn
        from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig
        if args.quant == "int8":
            cfg = Int8WeightOnlyConfig()
        else:
            # int4: 用 tinygemm 的 tile_packed_to_4d packing (aten::_convert_weight_to_int4pack)。
            # 这是 Ampere sm_80+ 原生路径, Orin 兼容; torchao 0.17 的默认 packing 需非便携的 mslk kernel。
            cfg = Int4WeightOnlyConfig(group_size=128, int4_packing_format="tile_packed_to_4d")
        # 排除 lm_head: Qwen tie_word_embeddings=True, 量化 tied lm_head 会打破与 embedding 的
        # 权重共享, 凭空新增一份量化副本(int8 +~136MB), 抵消一半收益。保持 lm_head bf16(与 embedding 共享)。
        quantize_(model.llm, cfg,
                  filter_fn=lambda mod, fqn: isinstance(mod, nn.Linear) and "lm_head" not in fqn)
        print(f"[quant] LLM 已量化: {args.quant} (torchao weight-only, 排除 tied lm_head)")

    # 权重常驻显存 (用于量化体积对比)
    resident_mb = None
    if device == "cuda":
        torch.cuda.empty_cache(); torch.cuda.synchronize()
        resident_mb = torch.cuda.memory_allocated() / 1024 ** 2
        print(f"[vram] 权重常驻显存: {resident_mb:.0f} MB")

    # 加载 val
    by_image = load_val(args.data)
    all_images = sorted(by_image.keys())
    if args.max_samples and args.max_samples < len(all_images):
        all_images = all_images[:args.max_samples]
    print(f"[data] 评测 {len(all_images)} 张图 ({sum(len(by_image[i]) for i in all_images)} 条 GT caption)")

    # 生成 caption + 收集
    prompt = (
        f"<|im_start|>user\n{IMAGE_TOKEN}\nDescribe this image.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    image_root = Path(args.image_root)
    results = []  # list of {image, gen, refs, bleu, gen_len}
    t0 = time.time()
    for idx, img_name in enumerate(all_images):
        img_path = image_root / img_name
        if not img_path.exists():
            print(f"[warn] 跳过缺失图: {img_name}")
            continue
        image = Image.open(img_path).convert("RGB")
        pixel_values = model.vision_encoder.image_processor(
            images=image, return_tensors="pt"
        )["pixel_values"].to(device)

        gen_out = model.generate(
            pixel_values=pixel_values,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,  # greedy for reproducibility
        )
        gen_text = gen_out["clean"] if isinstance(gen_out, dict) else gen_out

        refs = by_image[img_name]
        gen_tokens = tokenize(gen_text)
        refs_tokens = [tokenize(r) for r in refs]
        bleu = sentence_bleu4(gen_tokens, refs_tokens)

        results.append({
            "image": img_name,
            "gen": gen_text,
            "refs": refs,
            "bleu": bleu,
            "gen_len": len(gen_tokens),
            "ref_len_avg": sum(len(r) for r in refs_tokens) / len(refs_tokens),
        })

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx + 1}/{len(all_images)}] avg BLEU-4 so far: "
                  f"{sum(r['bleu'] for r in results) / len(results):.4f}  "
                  f"({elapsed:.0f}s)")

    # 汇总
    n = len(results)
    avg_bleu = sum(r["bleu"] for r in results) / max(n, 1)
    avg_gen_len = sum(r["gen_len"] for r in results) / max(n, 1)
    avg_ref_len = sum(r["ref_len_avg"] for r in results) / max(n, 1)
    # 标准 corpus BLEU-4 (与论文同法, 可对标)
    corpus_b, corpus_prec = corpus_bleu4(results)

    print()
    print("=" * 60)
    print(f"{'Baseline (未训)' if args.baseline else 'Trained projector'}")
    print("=" * 60)
    print(f"评测样本数:            {n}")
    print(f"corpus BLEU-4 (标准):  {corpus_b:.4f}   ({corpus_b * 100:.2f}%)  ← 与论文对标用")
    print(f"  n-gram 精度:         {[round(p, 3) for p in corpus_prec]}")
    print(f"sentence BLEU-4 (平滑): {avg_bleu:.4f}   ({avg_bleu * 100:.2f}%)  ← 内部进度追踪")
    print(f"平均生成长度:          {avg_gen_len:.1f} tokens")
    print(f"平均参考长度:          {avg_ref_len:.1f} tokens")
    print(f"总耗时:                {time.time() - t0:.1f}s")

    # 打印 sample
    if args.show_samples > 0:
        print()
        print(f"Sample 输出 (前 {args.show_samples} 条):")
        for r in results[:args.show_samples]:
            print(f"\n  🖼  {r['image']}  (BLEU-4={r['bleu']:.3f})")
            print(f"     GEN: {r['gen']}")
            print(f"     GT#1: {r['refs'][0]}")

    # 存 JSON
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "mode": "baseline" if args.baseline else "trained",
                "ckpt": str(args.ckpt) if not args.baseline else None,
                "quant": args.quant,
                "weights_resident_mb": resident_mb,
                "num_samples": n,
                "corpus_bleu4": corpus_b,
                "corpus_bleu4_precisions": corpus_prec,
                "avg_sentence_bleu4_smoothed": avg_bleu,
                "avg_gen_len": avg_gen_len,
                "avg_ref_len": avg_ref_len,
                "samples": results,
            }, f, ensure_ascii=False, indent=2)
        print(f"[save] 详细结果 → {args.out}")


if __name__ == "__main__":
    main()
