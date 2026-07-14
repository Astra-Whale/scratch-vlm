"""
Flickr8k (jxie/flickr8k, parquet) 数据准备

从 parquet(内嵌图像 bytes)解出图像 + caption,按标准 split 生成 JSONL:
  data/flickr8k/{train,val,test}.jsonl  ({"image","caption"})
  data/flickr8k/images/*.jpg

标准 Flickr8k split:~6000 train / 1000 val / 1000 test,每图 5 caption。
在 train 上训练、在**标准 test** 上评测 → 方法学正确、可对标(修掉 Flickr30k-test 上训练的问题)。

前置:data/flickr8k/*.parquet 已下载(curl jxie/flickr8k)。
用法:python data/prepare_flickr8k.py
"""
import io
import sys
import json
import hashlib
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

HERE = Path(__file__).parent
DIR = HERE / "flickr8k"
IMAGES_DIR = DIR / "images"

SPLIT_FILES = {
    "test": ["test-00000-of-00001-42a2661d12c73e48.parquet"],
    "val": ["validation-00000-of-00001-7025a2b596f14b7b.parquet"],
    "train": ["train-00000-of-00002-2f8f6bfa852eac4b.parquet",
              "train-00001-of-00002-2173151d8cd6c7fb.parquet"],
}


def detect_cols(tbl):
    """自动检测 image 列与全部 caption 列(Flickr8k 为 caption_0..caption_4 五列)。"""
    img_col = None
    cap_cols = []
    row0 = {c: tbl.column(c)[0].as_py() for c in tbl.column_names}
    for c, v in row0.items():
        if img_col is None and (isinstance(v, dict) and "bytes" in v or isinstance(v, (bytes, bytearray))):
            img_col = c
        elif isinstance(v, str) and (c.lower().startswith("caption") or c.lower() in ("text", "sentence")):
            cap_cols.append(c)
    return img_col, cap_cols


def img_bytes(v):
    return v["bytes"] if isinstance(v, dict) else v


def prepare_split(split, files):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    seen = {}  # hash -> filename
    n_img = n_pair = 0
    with open(DIR / f"{split}.jsonl", "w", encoding="utf-8") as fout:
        for fn in files:
            path = DIR / fn
            if not path.exists():
                print(f"[warn] 缺 {fn}, 跳过"); continue
            tbl = pq.read_table(path)
            img_col, cap_cols = detect_cols(tbl)
            if not img_col or not cap_cols:
                print(f"[error] {fn} 检测列失败: cols={tbl.column_names}"); sys.exit(1)
            imgs = tbl.column(img_col).to_pylist()
            caps_by_col = {c: tbl.column(c).to_pylist() for c in cap_cols}
            for i, iv in enumerate(imgs):
                b = img_bytes(iv)
                h = hashlib.md5(b).hexdigest()
                if h not in seen:
                    name = f"{h}.jpg"
                    Image.open(io.BytesIO(b)).convert("RGB").save(IMAGES_DIR / name)
                    seen[h] = name
                    n_img += 1
                name = seen[h]
                for c in cap_cols:
                    cap = (caps_by_col[c][i] or "").strip()
                    if cap:
                        fout.write(json.dumps({"image": name, "caption": cap}, ensure_ascii=False) + "\n")
                        n_pair += 1
    print(f"[{split:5}] {n_img:5d} 图, {n_pair:6d} 对 (image/caption 列: {img_col}/{cap_cols})")


if __name__ == "__main__":
    for split, files in SPLIT_FILES.items():
        prepare_split(split, files)
    print("\n[done] 训练 + 标准 test 评测:")
    print("  python train.py --data data/flickr8k/train.jsonl --val-data data/flickr8k/val.jsonl \\")
    print("     --image-root data/flickr8k/images --vision openai/clip-vit-large-patch14-336 \\")
    print("     --llm Qwen/Qwen2.5-0.5B-Instruct --init-projector checkpoints/projector_L14_qwenInstruct_ft_best.pt \\")
    print("     --steps 2000 --batch 4 --grad-accum 4 --lr 2e-4 --warmup 20 --val-every 100 \\")
    print("     --out checkpoints/projector_flickr8k.pt")
    print("  python evaluate.py --ckpt checkpoints/projector_flickr8k.pt \\")
    print("     --data data/flickr8k/test.jsonl --image-root data/flickr8k/images --max-samples 1000")
