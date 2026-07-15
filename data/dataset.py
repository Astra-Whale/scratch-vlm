"""
MVP Caption Dataset · 通用图文对 loader

数据格式 (JSONL, 每行):
    {"image": "path/to/image.png", "caption": "A caption."}
image 可为相对路径 (相对于指定的 image_root) 或绝对路径。

单样本构造流程:
    1. 加载图 → CLIP image_processor → pixel_values [3, 224, 224]
    2. 构造 prompt: "<image>\n{QUESTION}"  例如 "<image>\nDescribe this image."
    3. tokenize prompt (前半, label=-100) 和 answer (=caption, label=真实 token id)
    4. 拼成 full_ids + full_labels
"""
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image


# 默认 prompt (与 test_forward.py 一致)
DEFAULT_QUESTION = "Describe this image."


class MVPCaptionDataset(Dataset):
    """图文对数据集, 用于训 projector。"""

    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        image_processor,
        tokenizer,
        image_token: str = "<image>",
        question: str = DEFAULT_QUESTION,
        max_length: int = 128,
    ):
        self.samples = self._load_jsonl(jsonl_path)
        self.image_root = Path(image_root)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.question = question
        self.max_length = max_length

    @staticmethod
    def _load_jsonl(path: str) -> List[Dict[str, Any]]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        if p.is_absolute() or p.exists():
            return p
        return self.image_root / rel_or_abs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # 1. 图像 → pixel_values
        img_path = self._resolve_image_path(sample["image"])
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        )["pixel_values"][0]  # [3, H, W]

        # 2. 构造 prompt (前半, 不算 loss) 和 answer (=caption, 算 loss)
        # 使用 ChatML 格式, Instruct 模型必须走 chat 结构否则会秒 EOS
        prompt = (
            f"<|im_start|>user\n{self.image_token}\n{self.question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        answer = f"{sample['caption']}<|im_end|>"

        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=False
        )["input_ids"]
        answer_ids = self.tokenizer(
            answer, add_special_tokens=False
        )["input_ids"]

        full_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + list(answer_ids)

        # 截断保护
        if len(full_ids) > self.max_length:
            full_ids = full_ids[: self.max_length]
            labels = labels[: self.max_length]

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int):
    """把不定长样本 pad 成 batch。

    Returns dict:
        pixel_values: [B, 3, H, W]
        input_ids:    [B, T]
        attention_mask: [B, T]
        labels:       [B, T]  pad 处为 -100
    """
    B = len(batch)
    max_len = max(x["input_ids"].shape[0] for x in batch)

    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)

    pixel_values = torch.stack([x["pixel_values"] for x in batch], dim=0)

    for i, x in enumerate(batch):
        L = x["input_ids"].shape[0]
        input_ids[i, :L] = x["input_ids"]
        attention_mask[i, :L] = 1
        labels[i, :L] = x["labels"]

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
