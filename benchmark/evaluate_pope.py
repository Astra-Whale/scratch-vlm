"""
POPE 幻觉评测 (object-existence Yes/No) · 方法对齐 spec

POPE: 每条 {question_id, image, text(问句 "Is there a <obj> in the image?"), label(yes/no)}。
三 split: random / popular / adversarial(负样本采样策略不同)。
指标: Accuracy / Precision / Recall / F1 + yes 占比(检测过度肯定=幻觉倾向)。

支持对比 stage-1(仅 projector)vs stage-2(projector+LoRA), 量化两阶段对幻觉抑制的边际收益。

用法:
  # stage-1
  python benchmark/evaluate_pope.py --projector-ckpt checkpoints/projector_stage1_qwen3.pt \
      --pope-dir data/pope --image-root data/coco/val2014
  # stage-2 (加 LoRA)
  python benchmark/evaluate_pope.py --projector-ckpt checkpoints/vlm_stage2_lora/projector.pt \
      --lora-adapter checkpoints/vlm_stage2_lora/lora_adapter --pope-dir data/pope --image-root data/coco/val2014
"""
import os, sys, json, argparse
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from PIL import Image

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
from model.vlm import ScratchVLM, IMAGE_TOKEN

SPLITS = ["random", "popular", "adversarial"]
# 两种 prompt 口径:
#   single-word: 加 "Answer using a single word" 引导(稳定短答, 本项目旗舰口径)
#   original   : POPE 原论文口径, 只给问句, 不加引导
PROMPTS = {
    "single-word": ("<|im_start|>user\n{img}\n{q}\nAnswer the question using a single word, yes or no.<|im_end|>\n"
                    "<|im_start|>assistant\n"),
    "original": ("<|im_start|>user\n{img}\n{q}<|im_end|>\n"
                 "<|im_start|>assistant\n"),
}


def parse_yesno(text):
    """返回 (答案, 是否可解析)。无法从输出判定时保守判 no, 但标记 unparseable=True。"""
    t = text.strip().lower()
    if t.startswith("yes"): return "yes", True
    if t.startswith("no"): return "no", True
    if "yes" in t[:20] and "no" not in t[:20]: return "yes", True
    if "no" in t[:20] and "yes" not in t[:20]: return "no", True
    return "no", False  # 无法判定按 no(保守), 计入 unparseable


def f1_stats(preds, gts):
    tp = sum(p == "yes" and g == "yes" for p, g in zip(preds, gts))
    fp = sum(p == "yes" and g == "no" for p, g in zip(preds, gts))
    tn = sum(p == "no" and g == "no" for p, g in zip(preds, gts))
    fn = sum(p == "no" and g == "yes" for p, g in zip(preds, gts))
    n = len(preds)
    acc = (tp + tn) / max(1, n)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    yes_ratio = sum(p == "yes" for p in preds) / max(1, n)
    return {"acc": round(acc*100, 2), "precision": round(prec*100, 2), "recall": round(rec*100, 2),
            "f1": round(f1*100, 2), "yes_ratio": round(yes_ratio*100, 2), "n": n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--projector-ckpt", required=True)
    p.add_argument("--lora-adapter", default=None)
    p.add_argument("--pope-dir", default="data/pope")
    p.add_argument("--image-root", default="data/coco/val2014")
    p.add_argument("--max-per-split", type=int, default=0, help="0=全部")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--prompt-style", default="single-word", choices=["single-word", "original"],
                   help="single-word=加短答引导(旗舰口径);original=POPE 原论文只给问句")
    p.add_argument("--out", default="logs/pope_eval.json")
    args = p.parse_args()
    PROMPT = PROMPTS[args.prompt_style]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    ck = torch.load(args.projector_ckpt, map_location="cpu", weights_only=False)
    model = ScratchVLM(ck.get("vision_name", "openai/clip-vit-large-patch14-336"),
                       ck.get("llm_name", "Qwen/Qwen3-0.6B"), dtype=dtype, device=device).to(device)
    model.projector.load_state_dict(ck["projector_state_dict"])
    if args.lora_adapter:
        from peft import PeftModel
        model.llm = PeftModel.from_pretrained(model.llm, args.lora_adapter)
        print(f"[lora] loaded {args.lora_adapter}")
    model.eval()

    img_root = Path(args.image_root)
    results = {}
    for split in SPLITS:
        fp = Path(args.pope_dir) / f"coco_pope_{split}.json"
        if not fp.exists():
            print(f"[skip] 缺 {fp}"); continue
        rows = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
        if args.max_per_split:
            rows = rows[:args.max_per_split]
        preds, gts, miss, unpar = [], [], 0, 0
        for i, r in enumerate(rows):
            ipath = img_root / Path(r["image"]).name
            if not ipath.exists():
                miss += 1; continue
            image = Image.open(ipath).convert("RGB")
            pv = model.vision_encoder.image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
            g = model.generate(pixel_values=pv, prompt=PROMPT.format(img=IMAGE_TOKEN, q=r["text"]),
                               max_new_tokens=8, temperature=0.0)
            ans, ok = parse_yesno(g["clean"] if isinstance(g, dict) else g)
            preds.append(ans); unpar += (0 if ok else 1)
            gts.append(r["label"].strip().lower())
            if (i+1) % 500 == 0: print(f"  {split} {i+1}/{len(rows)}")
        st = f1_stats(preds, gts); st["missing_img"] = miss
        st["unparseable"] = unpar
        st["unparseable_ratio"] = round(unpar / max(1, len(preds)) * 100, 2)
        results[split] = st
        print(f"[{split:12}] acc={st['acc']} f1={st['f1']} yes%={st['yes_ratio']} (n={st['n']}, miss={miss})")

    if results:
        results["avg_f1"] = round(sum(results[s]["f1"] for s in results if s in SPLITS) / len([s for s in results if s in SPLITS]), 2)
        print(f"[avg F1] {results['avg_f1']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
