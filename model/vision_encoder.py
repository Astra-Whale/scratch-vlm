"""
CLIP-ViT-L/14 视觉编码器

- 加载 HuggingFace 的 openai/clip-vit-large-patch14
- 冻结所有参数(VLM 训练时不参与更新)
- 输出 patch-level 视觉特征, 供 projector 映射到 LLM embedding 空间

设计决策:
- select_layer=-2: 参考 LLaVA v1.5, 选倒数第二层特征效果更好
  (最后一层过于向语言侧塌缩, 丢失视觉细节)
- select_feature="patch": 去掉 CLS token, 保留 576 个 patch token
"""
# 默认走 hf-mirror(与 vlm.py 一致);走官方站请 export HF_ENDPOINT=https://huggingface.co
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from typing import Optional
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor


class VisionEncoder(nn.Module):
    """CLIP-ViT-L/14 视觉塔(冻结)。"""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        select_layer: int = -2,
        select_feature: str = "patch",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.model_name = model_name
        self.select_layer = select_layer
        self.select_feature = select_feature
        self.dtype = dtype

        # 加载 CLIP 视觉塔与其对应的 image processor
        self.vision_tower = CLIPVisionModel.from_pretrained(
            model_name, torch_dtype=dtype
        )
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)

        # 冻结所有参数, 训练时不参与更新
        for p in self.vision_tower.parameters():
            p.requires_grad = False
        self.vision_tower.eval()

    @property
    def hidden_size(self) -> int:
        """CLIP-ViT-L/14 hidden_size = 1024"""
        return self.vision_tower.config.hidden_size

    @property
    def num_patches(self) -> int:
        """在 224x224 输入下 patch=14, 产生 (224/14)**2 = 16**2 = 256 个 patch。

        注: 之前文档写 576 是笔误——CLIP-ViT-L/14 官方 image_size=224, patch_size=14,
        实际 patch 数是 16x16=256; 若换成 336px 输入 (LLaVA v1.5 训练分辨率),
        才是 (336/14)**2 = 24*24 = 576。此处以配置为准。
        """
        cfg = self.vision_tower.config
        return (cfg.image_size // cfg.patch_size) ** 2

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """前向: 像素 → patch 特征。

        Args:
            pixel_values: [B, 3, H, W], H=W=224 (openai/clip-vit-large-patch14 默认)

        Returns:
            features: [B, num_patches, hidden_size], 默认 [B, 256, 1024]
        """
        outputs = self.vision_tower(
            pixel_values.to(dtype=self.dtype),
            output_hidden_states=True,
        )
        # 选取指定层的 hidden states
        features = outputs.hidden_states[self.select_layer]  # [B, 1+num_patches, D]

        # 是否去掉 CLS token
        if self.select_feature == "patch":
            features = features[:, 1:]  # 去掉位置 0 的 CLS
        elif self.select_feature == "cls_patch":
            pass
        else:
            raise ValueError(f"未知 select_feature: {self.select_feature}")

        return features

    def preprocess(self, images):
        """PIL Image 列表 → pixel_values tensor [B, 3, 224, 224]。"""
        return self.image_processor(images=images, return_tensors="pt")["pixel_values"]
