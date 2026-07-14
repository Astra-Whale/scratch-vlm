"""
按 image 划分 Flickr1K → train / val

避免 caption-level 划分导致的 leakage:
    每张图 5 条 caption, 若按 caption 划就会把同一张图的部分 caption
    分到 val, 那 val 上的 loss 反映的是 image 记忆而非 generalization。

按 image 层面稳定划分:
    - 1000 unique images
    - 前 900 (按 filename 排序) → train (~4500 pairs)
    - 后 100 → val (~500 pairs)

用法:
    python data/split_flickr.py
"""
import os
import sys
import json
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = Path(__file__).parent
FLICKR = HERE / "flickr_1k"
FULL_JSONL = FLICKR / "flickr_1k.jsonl"
TRAIN_JSONL = FLICKR / "flickr_1k_train.jsonl"
VAL_JSONL = FLICKR / "flickr_1k_val.jsonl"

N_VAL_IMAGES = 100  # 100 张图作 val, 900 张作 train


def main():
    if not FULL_JSONL.exists():
        print(f"[error] 未找到 {FULL_JSONL}, 请先跑 prepare_flickr.py")
        sys.exit(1)

    # 读全部
    records = []
    with open(FULL_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # 按 image 分桶
    images = sorted(set(r["image"] for r in records))
    print(f"[split] 总样本: {len(records)}, unique 图: {len(images)}")
    assert len(images) >= N_VAL_IMAGES * 2, "图太少, 划分不合理"

    # 稳定按 filename 排序后取���部 100 张作 val
    val_images = set(images[-N_VAL_IMAGES:])
    train_images = set(images[:-N_VAL_IMAGES])
    assert val_images.isdisjoint(train_images), "train/val image 重叠"

    train_rows = [r for r in records if r["image"] in train_images]
    val_rows = [r for r in records if r["image"] in val_images]

    print(f"[split] train: {len(train_images)} imgs, {len(train_rows)} pairs")
    print(f"[split] val:   {len(val_images)} imgs, {len(val_rows)} pairs")

    with open(TRAIN_JSONL, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(VAL_JSONL, "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[write] {TRAIN_JSONL}")
    print(f"[write] {VAL_JSONL}")


if __name__ == "__main__":
    main()
