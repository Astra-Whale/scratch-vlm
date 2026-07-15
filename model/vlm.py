"""
散装 VLM 拼装 (LLaVA v1.5 风格)

关键设计:
- 冻结 CLIP 视觉塔 (~304M 参数)
- 冻结 Qwen 语言模型 (~500M 参数)
- 只训练 MLP Projector (~4M 参数)

<image> 特殊 token 用于在 prompt 中占位, 训练/推理时被替换为
视觉特征投影后的 embedding, 与文本 token embedding 拼接后送入 LLM。
"""
# 中国大陆直连 HuggingFace 速度慢, 默认使用 hf-mirror 加速。
# 如需强制走原站, 请在启动前设置 HF_ENDPOINT=https://huggingface.co
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .vision_encoder import VisionEncoder
from .projector import MLPProjector


# 用于在 prompt 中占位图像位置的特殊 token
IMAGE_TOKEN = "<image>"


class ScratchVLM(nn.Module):
    """散装 VLM: CLIP-ViT + MLP Projector + Qwen LLM。"""

    def __init__(
        self,
        vision_model_name: str = "openai/clip-vit-large-patch14",
        llm_model_name: str = "Qwen/Qwen3-0.6B",
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        super().__init__()
        self.dtype = dtype
        self.device_str = device

        # ============ 1. 视觉塔 (冻结) ============
        self.vision_encoder = VisionEncoder(vision_model_name, dtype=dtype)

        # ============ 2. 语言塔 (冻结) ============
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, torch_dtype=dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

        # 冻结 LLM 所有参数
        for p in self.llm.parameters():
            p.requires_grad = False
        self.llm.eval()

        # ============ 3. Projector (唯一可训层) ============
        # 从 LLM config 自动读 hidden_size, 保持维度对齐
        llm_hidden = self.llm.config.hidden_size
        self.projector = MLPProjector(
            input_dim=self.vision_encoder.hidden_size,
            hidden_dim=2048,
            output_dim=llm_hidden,
            dtype=dtype,
        )

        # ============ 4. 添加 <image> 特殊 token ============
        vocab = self.tokenizer.get_vocab()
        if IMAGE_TOKEN not in vocab:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [IMAGE_TOKEN]}
            )
            self.llm.resize_token_embeddings(len(self.tokenizer))
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    # ==============================================================
    # 图像编码 (VLM 前向核心之一)
    # ==============================================================
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """图像像素 → LLM embedding-space 上的 visual tokens。

        Args:
            pixel_values: [B, 3, 224, 224]

        Returns:
            visual_tokens: [B, N_v, llm_hidden]
        """
        with torch.no_grad():
            visual = self.vision_encoder(pixel_values)  # [B, N_v, vision_dim]
        projected = self.projector(visual)              # [B, N_v, llm_dim]
        return projected

    # ==============================================================
    # 训练前向 (与 HF Trainer 兼容的 signature)
    # ==============================================================
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """训练前向。

        流程:
          1. 用 vision_encoder + projector 得到 visual_tokens
          2. 把 input_ids 里 <image> token 位置替换为 visual_tokens embedding
          3. 送入 LLM 计算 next-token loss (visual tokens 位置 label=-100)

        每条样本必须有 恰好一个 <image> token 占位。

        Args:
            pixel_values: [B, 3, 224, 224]
            input_ids: [B, T]  必须包含 <image> token
            attention_mask: [B, T]
            labels: [B, T]  可为 None (只做前向), 视觉位置会被替换为 -100
        """
        B = input_ids.shape[0]

        # 1. 视觉 tokens
        visual_tokens = self.encode_images(pixel_values)  # [B, N_v, D]
        N_v = visual_tokens.shape[1]

        # 2. 文本 tokens 的 embedding
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, N_t, D]
        D = text_embeds.shape[-1]

        # 3. 逐样本替换 <image> 位置为 visual tokens
        new_embeds_list = []
        new_labels_list = []
        new_mask_list = []

        for b in range(B):
            img_positions = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]

            if len(img_positions) == 0:
                # 无图像 token: 直接文本
                new_embeds_list.append(text_embeds[b])
                if labels is not None:
                    new_labels_list.append(labels[b])
                if attention_mask is not None:
                    new_mask_list.append(attention_mask[b])
                continue

            p = img_positions[0].item()

            # embedding 拼接: [文本前段] + [visual tokens] + [文本后段(去掉 <image>)]
            pre = text_embeds[b, :p]
            post = text_embeds[b, p + 1:]
            new_emb = torch.cat([pre, visual_tokens[b], post], dim=0)
            new_embeds_list.append(new_emb)

            # labels 对齐: visual 位置置 -100 (不算 loss)
            if labels is not None:
                pre_l = labels[b, :p]
                post_l = labels[b, p + 1:]
                vis_l = torch.full(
                    (N_v,), -100, dtype=labels.dtype, device=labels.device
                )
                new_labels_list.append(torch.cat([pre_l, vis_l, post_l]))

            # attention_mask 对齐: visual 位置置 1
            if attention_mask is not None:
                pre_m = attention_mask[b, :p]
                post_m = attention_mask[b, p + 1:]
                vis_m = torch.ones(
                    N_v, dtype=attention_mask.dtype, device=attention_mask.device
                )
                new_mask_list.append(torch.cat([pre_m, vis_m, post_m]))

        # 4. batch padding (各样本替换后长度不同)
        max_len = max(x.shape[0] for x in new_embeds_list)

        padded_embeds = torch.zeros(
            B, max_len, D, dtype=self.dtype, device=text_embeds.device
        )
        padded_labels = (
            torch.full(
                (B, max_len), -100, dtype=torch.long, device=text_embeds.device
            )
            if labels is not None else None
        )
        padded_mask = torch.zeros(
            B, max_len, dtype=torch.long, device=text_embeds.device
        )

        for b in range(B):
            L = new_embeds_list[b].shape[0]
            padded_embeds[b, :L] = new_embeds_list[b]
            if padded_labels is not None and b < len(new_labels_list):
                padded_labels[b, :L] = new_labels_list[b]
            if b < len(new_mask_list):
                padded_mask[b, :L] = new_mask_list[b]
            else:
                padded_mask[b, :L] = 1

        # 5. 送入 LLM
        return self.llm(
            inputs_embeds=padded_embeds,
            attention_mask=padded_mask,
            labels=padded_labels,
        )

    # ==============================================================
    # 推理接口
    # ==============================================================
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> str:
        """单图推理: 图像 + 文本 prompt → 生成回答。

        prompt 里必须包含 IMAGE_TOKEN 占位符, 例如:
            "<image>\n描述这张图片。"
        """
        device = self.llm.device

        # tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]  # [1, T]

        # 视觉 tokens
        visual_tokens = self.encode_images(pixel_values.to(device))  # [1, N_v, D]

        # 文本 embedding + 拼接
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # [1, T, D]
        img_positions = (input_ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]

        if len(img_positions) == 0:
            raise ValueError(f"prompt 必须包含 IMAGE_TOKEN='{IMAGE_TOKEN}'")

        p = img_positions[0].item()
        pre = text_embeds[0, :p]
        post = text_embeds[0, p + 1:]
        embeds = torch.cat([pre, visual_tokens[0], post], dim=0).unsqueeze(0)  # [1, T', D]
        attn = torch.ones(embeds.shape[:2], dtype=torch.long, device=device)

        # 生成
        # ChatML 回合结束符 <|im_end|>: 部分 base 模型默认 eos 只有 <|endoftext|>,
        # 不含 <|im_end|>。不显式设置会导致生成越过 caption 继续吐 token, 拖低 BLEU。
        stop_ids = []
        if self.tokenizer.eos_token_id is not None:
            stop_ids.append(self.tokenizer.eos_token_id)
        _im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(_im_end, int) and _im_end >= 0 and _im_end not in stop_ids:
            stop_ids.append(_im_end)

        gen_kwargs = dict(
            inputs_embeds=embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=stop_ids or None,
        )
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=max(temperature, 1e-4))
        else:
            gen_kwargs.update(do_sample=False)  # greedy

        output_ids = self.llm.generate(**gen_kwargs)
        # 传 inputs_embeds 时, output_ids 只含新生成的 tokens (不含 prompt)
        raw_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        clean_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Qwen3 thinking mode 可能自发生成 <think>...</think>, 剥离掉只留最终答案
        import re
        clean_text = re.sub(r"<think>.*?</think>", "", clean_text, flags=re.DOTALL).strip()
        # 返回 (raw + clean) 便于调试; 后续可换成只返 clean
        return {
            "raw": raw_text,
            "clean": clean_text,
            "output_ids": output_ids[0].tolist(),
            "num_new_tokens": output_ids.shape[1],
        }

    # ==============================================================
    # 便利方法
    # ==============================================================
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
