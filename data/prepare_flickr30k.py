"""
完整 Flickr30k 数据准备脚本 (Karpathy split)

与 prepare_flickr.py(仅 1k test 子集)不同:本脚本处理**完整 Flickr30k**,
按 CSV 的 `split` 列拆成 train / val / test 三份 JSONL。

- train (~29k 图) 用于训练
- test  (1000 图, 即原 flickr_1k 那批) 用于评测 —— **标准 Karpathy test split, 可对标论文**

前置(hf download nlphuji/flickr30k → data/flickr30k/):
  - flickr_annotations_30k.csv
  - flickr30k-images.zip (~4.4 GB)

用法:
  python data/prepare_flickr30k.py

输出:
  data/flickr30k/images/*.jpg
  data/flickr30k/{train,val,test}.jsonl   (每行 {"image","caption"})
"""
import sys
import csv
import json
import zipfile
from pathlib import Path
from collections import defaultdict

csv.field_size_limit(10 ** 7)

HERE = Path(__file__).parent
DIR = HERE / "flickr30k"
CSV_PATH = DIR / "flickr_annotations_30k.csv"
ZIP_PATH = DIR / "flickr30k-images.zip"
IMAGES_DIR = DIR / "images"


def unzip_images(force: bool = False):
    if IMAGES_DIR.exists() and not force and len(list(IMAGES_DIR.glob("*.jpg"))) > 30000:
        print(f"[unzip] images/ 已有 {len(list(IMAGES_DIR.glob('*.jpg')))} 张, 跳过")
        return
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[unzip] {ZIP_PATH.name} → {IMAGES_DIR} (~31k 图, 稍候)")
    n, skipped = 0, 0
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for m in zf.namelist():
            if m.endswith("/") or not m.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            base = Path(m).name
            if "__MACOSX" in m or base.startswith("._"):
                skipped += 1
                continue
            with zf.open(m) as src, open(IMAGES_DIR / base, "wb") as dst:
                dst.write(src.read())
            n += 1
    print(f"[unzip] 完成 {n} 张 (跳过 {skipped} 元数据)")


def csv_to_splits():
    """按 split 列拆分, 展平 raw(5 caption) → {split}.jsonl。"""
    counts_img = defaultdict(int)
    counts_pair = defaultdict(int)
    writers = {s: open(DIR / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "val", "test")}
    with open(CSV_PATH, "r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            split = (row.get("split") or "").strip()
            if split not in writers:
                continue
            filename = row["filename"]
            try:
                captions = json.loads(row["raw"])
            except json.JSONDecodeError:
                continue
            counts_img[split] += 1
            for cap in captions:
                cap = cap.strip()
                if cap:
                    writers[split].write(json.dumps({"image": filename, "caption": cap}, ensure_ascii=False) + "\n")
                    counts_pair[split] += 1
    for w in writers.values():
        w.close()
    for s in ("train", "val", "test"):
        print(f"[jsonl] {s:5}: {counts_img[s]:6d} 图, {counts_pair[s]:7d} 对 → {DIR / (s + '.jsonl')}")


if __name__ == "__main__":
    if not CSV_PATH.exists() or not ZIP_PATH.exists():
        print(f"[error] 缺文件, 先: hf download nlphuji/flickr30k flickr_annotations_30k.csv flickr30k-images.zip --repo-type dataset --local-dir {DIR}")
        sys.exit(1)
    unzip_images()
    csv_to_splits()
    print("\n[done] 训练(train)+ 标准 test 评测:")
    print("  python train.py --data data/flickr30k/train.jsonl --val-data data/flickr30k/val.jsonl \\")
    print("     --image-root data/flickr30k/images --vision openai/clip-vit-large-patch14-336 \\")
    print("     --llm Qwen/Qwen2.5-0.5B-Instruct --init-projector checkpoints/projector_L14_qwenInstruct_ft_best.pt \\")
    print("     --steps 2000 --batch 4 --grad-accum 4 --lr 2e-4 --warmup 20 --val-every 100 \\")
    print("     --out checkpoints/projector_flickr30k.pt")
    print("  python evaluate.py --ckpt checkpoints/projector_flickr30k.pt \\")
    print("     --data data/flickr30k/test.jsonl --image-root data/flickr30k/images --max-samples 1000")
