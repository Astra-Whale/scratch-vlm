"""
视觉指令微调 (stage-2 SFT) 数据集 · LLaVA-Instruct 对话格式

输入 JSON(LLaVA-Instruct 风格): [{"id", "image": "000000xxxxxx.jpg",
  "conversations": [{"from":"human","value":"...<image>..."}, {"from":"gpt","value":"..."}, ...]}]

构造(方法对齐 LLaVA v1.5 SFT):
  - ChatML 手工拼接(不走 apply_chat_template, 从而**天然规避 Qwen3 thinking `<think>` 注入**)
  - 首个 human turn 里的 `<image>` → 项目 IMAGE_TOKEN(forward 时替换为视觉 token)
  - **label masking: 只对 gpt/assistant turn 的内容(含其 `<|im_end|>`)算 loss**, 其余 -100
  - 与 ScratchVLM.forward 无缝: 产出 input_ids/labels/attention_mask, forward 负责 <image>→视觉token 替换
"""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

IMAGE_TOKEN = "<image>"


class LlavaInstructDataset(Dataset):
    def __init__(self, json_path, image_root, image_processor, tokenizer,
                 image_token=IMAGE_TOKEN, max_length=1024):
        self.samples = json.load(open(json_path, encoding="utf-8"))
        self.image_root = Path(image_root)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.max_length = max_length
        # 过滤掉本地缺图的样本(COCO 只抓了 subset)
        self.samples = [s for s in self.samples if (self.image_root / Path(s["image"]).name).exists()]

    def __len__(self):
        return len(self.samples)

    def _tok(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(self.image_root / Path(s["image"]).name).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]

        input_ids, labels = [], []
        convs = s["conversations"]
        for i in range(0, len(convs) - 1, 2):
            human = convs[i]["value"].strip()
            gpt = convs[i + 1]["value"].strip()
            # 只保留一个 <image> 占位(LLaVA 放在首个 human turn), 去掉多余换行
            human = human.replace("<image>", "").strip()
            if i == 0:
                human = f"{self.image_token}\n{human}"
            user_part = f"<|im_start|>user\n{human}<|im_end|>\n<|im_start|>assistant\n"
            gpt_part = f"{gpt}<|im_end|>\n"
            uids = self._tok(user_part)
            gids = self._tok(gpt_part)
            input_ids += uids + gids
            labels += [-100] * len(uids) + gids  # 只对 assistant 内容算 loss

        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]
        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
