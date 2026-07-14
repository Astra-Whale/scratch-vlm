"""
按需并行抓取 COCO 图(供 LLaVA-Instruct stage-2 SFT 与 POPE 评测配图)。

COCO 官方图床 images.cocodataset.org 直连 TLS 证书不匹配(CN=s3.amazonaws.com),
故一律走 **path-style S3**:
  https://s3.amazonaws.com/images.cocodataset.org/{split}/COCO_{split}_<12位id>.jpg

从一个列出图像名的 JSON 抽唯一图 → 并行下载 → 存到 out 目录(文件名与源 JSON 的
`image` 字段一致,便于 dataset.py 按 image-root 解析)。已存在则跳过,失败重试。

用法:
  # stage-2: 抽 detail_23k 前 N 张 train2014
  python data/fetch_coco_images.py --src data/llava_instruct/detail_23k.json \
      --field image --split train2014 --out data/coco/train2014 --max 3000 --workers 16
  # POPE: val2014 全部唯一图
  python data/fetch_coco_images.py --src data/pope/coco_pope_random.json --jsonl \
      --field image --split val2014 --out data/coco/val2014 --workers 16
"""
import os
import sys
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE = "https://s3.amazonaws.com/images.cocodataset.org"


def load_names(src, field, jsonl):
    names = []
    if jsonl:
        for line in open(src, encoding="utf-8"):
            line = line.strip()
            if line:
                names.append(json.loads(line)[field])
    else:
        for r in json.load(open(src, encoding="utf-8")):
            names.append(r[field])
    # 唯一 + 保序
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n); out.append(n)
    return out


def to_coco_url_and_name(name, split):
    """源里的 image 字段可能是 '000000442786.jpg' 或 'COCO_val2014_000000442786.jpg'。
    统一还原成 COCO 官方文件名构 URL;本地保存名保持与源一致。"""
    base = Path(name).name
    if base.startswith("COCO_"):
        coco_fn = base
    else:
        digits = "".join(c for c in base if c.isdigit())
        coco_fn = f"COCO_{split}_{int(digits):012d}.jpg"
    url = f"{BASE}/{split}/{coco_fn}"
    return url, base


def fetch_one(name, split, out_dir, retries=4):
    url, save_name = to_coco_url_and_name(name, split)
    dst = out_dir / save_name
    if dst.exists() and dst.stat().st_size > 0:
        return ("skip", save_name)
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and r.content:
                dst.write_bytes(r.content)
                return ("ok", save_name)
        except Exception:
            pass
    return ("fail", save_name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--field", default="image")
    p.add_argument("--jsonl", action="store_true", help="源是 JSONL(每行一条),否则 JSON 数组")
    p.add_argument("--split", required=True, choices=["train2014", "val2014"])
    p.add_argument("--out", required=True)
    p.add_argument("--max", type=int, default=0, help="0=全部")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    names = load_names(args.src, args.field, args.jsonl)
    if args.max:
        names = names[:args.max]
    print(f"[fetch] {len(names)} 张唯一图 → {out_dir} (workers={args.workers})")

    ok = skip = fail = 0
    fails = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, n, args.split, out_dir): n for n in names}
        for i, f in enumerate(as_completed(futs), 1):
            status, nm = f.result()
            if status == "ok": ok += 1
            elif status == "skip": skip += 1
            else: fail += 1; fails.append(nm)
            if i % 200 == 0:
                print(f"  {i}/{len(names)}  ok={ok} skip={skip} fail={fail}")
    print(f"[done] ok={ok} skip={skip} fail={fail}")
    if fails:
        print(f"[fail] {len(fails)} 张失败(前5): {fails[:5]}")
        Path(str(out_dir) + "_failed.txt").write_text("\n".join(fails))


if __name__ == "__main__":
    main()
