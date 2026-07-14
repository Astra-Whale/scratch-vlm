"""
Flickr1K 数据准备脚本

流程:
    1. 解压 images_flickr_1k_test.zip → data/flickr_1k/images/
    2. 解析 test_1k_flickr.csv 里的 raw 字段 (JSON array 5 条 caption)
    3. 展平为 5000 行 JSONL: {"image": "xxx.jpg", "caption": "..."}

前置:
    数据已下载至 data/flickr_1k/
      - test_1k_flickr.csv (1.1 MB)
      - images_flickr_1k_test.zip (134 MB)

用法:
    python data/prepare_flickr.py

输出:
    data/flickr_1k/images/*.jpg  (1000 张)
    data/flickr_1k/flickr_1k.jsonl (5000 行)
"""
import os
import sys
import csv
import json
import zipfile
from pathlib import Path

# Windows UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = Path(__file__).parent
FLICKR_DIR = HERE / "flickr_1k"
CSV_PATH = FLICKR_DIR / "test_1k_flickr.csv"
ZIP_PATH = FLICKR_DIR / "images_flickr_1k_test.zip"
IMAGES_DIR = FLICKR_DIR / "images"
JSONL_PATH = FLICKR_DIR / "flickr_1k.jsonl"


def unzip_images(force: bool = False):
    """解压 zip 到 images/ 目录 (若已解压且非 force 则跳过)。"""
    if IMAGES_DIR.exists() and not force:
        existing = list(IMAGES_DIR.glob("*.jpg"))
        if len(existing) > 900:
            print(f"[unzip] images/ 已有 {len(existing)} 张图, 跳过解压")
            return
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[unzip] {ZIP_PATH.name} → {IMAGES_DIR}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        # zip 里的路径结构可能带前缀 (如 flickr30k-images/xxx.jpg 或裸 xxx.jpg)
        members = zf.namelist()
        print(f"[unzip] zip 内 {len(members)} 个成员, 前几个: {members[:3]}")
        skipped_meta = 0
        for m in members:
            if m.endswith("/") or not m.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            # 过滤 macOS zip 里的 __MACOSX/._xxx.jpg 元数据 blob
            basename = Path(m).name
            if "__MACOSX" in m or basename.startswith("._"):
                skipped_meta += 1
                continue
            with zf.open(m) as src, open(IMAGES_DIR / basename, "wb") as dst:
                dst.write(src.read())
        if skipped_meta:
            print(f"[unzip] 跳过 {skipped_meta} 个 macOS 元数据文件")
    final = list(IMAGES_DIR.glob("*.jpg"))
    print(f"[unzip] 解压完成, 得到 {len(final)} 张图")


def csv_to_jsonl():
    """展平 CSV → 5000 行 JSONL。"""
    n_images = 0
    n_pairs = 0
    with open(CSV_PATH, "r", encoding="utf-8", newline="") as fin, \
         open(JSONL_PATH, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            filename = row["filename"]
            # raw 是 JSON 字符串: '["caption1", "caption2", ...]'
            try:
                captions = json.loads(row["raw"])
            except json.JSONDecodeError as e:
                print(f"[warn] 跳过 {filename}, raw 解析失败: {e}")
                continue
            n_images += 1
            for cap in captions:
                if not cap.strip():
                    continue
                fout.write(
                    json.dumps({"image": filename, "caption": cap.strip()},
                               ensure_ascii=False) + "\n"
                )
                n_pairs += 1
    print(f"[jsonl] {n_images} 张图, {n_pairs} 条 image-caption 对 → {JSONL_PATH}")


def sanity_check():
    """随机抽查, 确保图文对齐无问题。"""
    import random
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"[check] JSONL 共 {len(lines)} 行, 随机抽 3 条:")
    for line in random.sample(lines, min(3, len(lines))):
        rec = json.loads(line)
        img_path = IMAGES_DIR / rec["image"]
        status = "✓" if img_path.exists() else "✗ MISSING"
        print(f"  {status}  {rec['image']}: {rec['caption'][:80]}")


if __name__ == "__main__":
    if not CSV_PATH.exists():
        print(f"[error] CSV 不存在: {CSV_PATH}")
        print(f"       请先下载: hf-mirror.com/datasets/nlphuji/flickr_1k_test_image_text_retrieval")
        sys.exit(1)
    if not ZIP_PATH.exists():
        print(f"[error] ZIP 不存在: {ZIP_PATH}")
        sys.exit(1)

    unzip_images()
    csv_to_jsonl()
    sanity_check()

    print()
    print("[done] 可以开始训练了:")
    print("       python train.py \\")
    print(f"         --data {JSONL_PATH.relative_to(HERE.parent).as_posix()} \\")
    print(f"         --image-root {IMAGES_DIR.relative_to(HERE.parent).as_posix()} \\")
    print("         --steps 200 --batch 4 --lr 1e-3")
