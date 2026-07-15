"""
Toy Dataset · 合成 24 张彩色几何图 + caption

目的:
- Week 2 D2 里程碑仅验证训练闭环 (loss 是否下降), 不追求真实效果
- 用 PIL 生成小规模合成数据, 零下载, 秒级 setup
- 数据放在 data/toy_images/, JSONL 描述在 data/toy.jsonl

生成方式:
    python data/toy_dataset.py

之后训练:
    python train.py --data data/toy.jsonl --steps 30
"""
import os
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
IMAGES_DIR = HERE / "toy_images"
JSONL_PATH = HERE / "toy.jsonl"


# 8 种颜色 · 中英名对齐, 便于 LLM 拿到语义
COLORS = [
    ("red",     "红色", (220, 50, 50)),
    ("green",   "绿色", (50, 180, 80)),
    ("blue",    "蓝色", (60, 100, 220)),
    ("yellow",  "黄色", (240, 210, 50)),
    ("purple",  "紫色", (150, 80, 180)),
    ("orange",  "橙色", (240, 140, 50)),
    ("pink",    "粉色", (240, 150, 180)),
    ("cyan",    "青色", (80, 200, 210)),
]

# 3 种形状
SHAPES = ["square", "circle", "triangle"]
SHAPE_ZH = {"square": "方块", "circle": "圆", "triangle": "三角形"}


def draw_shape(shape: str, color_rgb, size: int = 224) -> Image.Image:
    """在 224×224 白底上画一个居中的彩色形状。"""
    img = Image.new("RGB", (size, size), (245, 245, 245))
    d = ImageDraw.Draw(img)
    m = size // 4  # 边距
    if shape == "square":
        d.rectangle([m, m, size - m, size - m], fill=color_rgb)
    elif shape == "circle":
        d.ellipse([m, m, size - m, size - m], fill=color_rgb)
    elif shape == "triangle":
        d.polygon(
            [(size // 2, m), (m, size - m), (size - m, size - m)],
            fill=color_rgb,
        )
    return img


def make_caption(color_en: str, color_zh: str, shape: str) -> str:
    """按 30% 概率中文 caption, 70% 英文 caption (LLM 侧偏英文)。"""
    shape_zh = SHAPE_ZH[shape]
    if random.random() < 0.3:
        return f"一个{color_zh}的{shape_zh}。"
    else:
        # 英文变体 (让 LLM 学到不同措辞)
        variants = [
            f"A {color_en} {shape}.",
            f"An image of a {color_en} {shape}.",
            f"A single {color_en} {shape} on a light background.",
        ]
        return random.choice(variants)


def build_toy_dataset(n_samples: int = 24, seed: int = 42):
    """生成 n_samples 张合成图 + JSONL 索引。"""
    random.seed(seed)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    combos = [(c, s) for c in COLORS for s in SHAPES]  # 24 组 = 8 * 3
    random.shuffle(combos)
    combos = combos[:n_samples]

    records = []
    for i, ((color_en, color_zh, rgb), shape) in enumerate(combos):
        img = draw_shape(shape, rgb, size=224)
        img_path = IMAGES_DIR / f"{i:03d}_{color_en}_{shape}.png"
        img.save(img_path)

        caption = make_caption(color_en, color_zh, shape)
        records.append({
            "image": img_path.name,     # 相对 data/toy_images/ 的文件名
            "caption": caption,
        })

    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[toy] 生成 {n_samples} 张图到 {IMAGES_DIR}")
    print(f"[toy] 索引写入 {JSONL_PATH}")
    print(f"[toy] 示例记录: {records[0]}")


if __name__ == "__main__":
    build_toy_dataset(n_samples=24)
