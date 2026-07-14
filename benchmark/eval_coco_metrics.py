"""
用 pycocoevalcap 官方实现,从已保存的评测 JSON 计算标准 COCO captioning 指标。

不重跑模型 —— 直接读 evaluate.py --out 存下的 samples(含 gen + refs),
用官方 PTBTokenizer 分词后跑 BLEU-1..4 / METEOR / ROUGE-L / CIDEr / SPICE。
这样得到与论文可对标的现代主指标(尤其 CIDEr),并规避自写 regex 分词的口径差异。

用法:
  python benchmark/eval_coco_metrics.py --json logs/eval_flickr8k_test_1000.json
"""
import sys
import json
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=str, required=True, help="evaluate.py --out 保存的结果 JSON")
    p.add_argument("--no-spice", action="store_true", help="跳过 SPICE(慢/吃内存)")
    p.add_argument("--out", type=str, default=None, help="指标结果 JSON 输出路径")
    return p.parse_args()


def main():
    args = parse_args()
    data = json.load(open(args.json, encoding="utf-8"))
    samples = data["samples"]

    # 构造 COCO 格式: {id: [{'caption': ...}]}
    gts, res = {}, {}
    for i, s in enumerate(samples):
        key = s.get("image", str(i))
        gts[key] = [{"caption": c} for c in s["refs"]]
        res[key] = [{"caption": s["gen"] if s["gen"].strip() else "."}]  # 空生成占位, 避免 tokenizer 报错

    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    print(f"[tok] PTBTokenizer 分词 {len(gts)} 图 ...")
    tok = PTBTokenizer()
    gts = tok.tokenize(gts)
    res = tok.tokenize(res)

    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    if not args.no_spice:
        try:
            from pycocoevalcap.spice.spice import Spice
            scorers.append((Spice(), "SPICE"))
        except Exception as e:
            print(f"[warn] SPICE 不可用, 跳过: {str(e)[:80]}")

    results = {}
    for scorer, name in scorers:
        try:
            score, _ = scorer.compute_score(gts, res)
        except Exception as e:
            print(f"[warn] {name} 失败: {str(e)[:100]}")
            continue
        if isinstance(name, list):
            for n, sc in zip(name, score):
                results[n] = round(sc * 100, 2) if n.startswith("Bleu") else round(sc, 4)
        else:
            # CIDEr 常以 100 为基准的尺度; METEOR/ROUGE/SPICE 在 0-1
            results[name] = round(score, 4)

    print("\n" + "=" * 56)
    print(f"标准 COCO 指标 · {Path(args.json).name} · {len(gts)} 图")
    print("=" * 56)
    for k, v in results.items():
        print(f"  {k:10} = {v}")
    print("\n注: BLEU_x 为 % (0-100); METEOR/ROUGE_L/SPICE 为 0-1; CIDEr 以 100 为基准的尺度。")

    out = args.out or args.json.replace(".json", "_cocometrics.json")
    json.dump(results, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
