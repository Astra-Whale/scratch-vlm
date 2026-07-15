"""
MLP Projector (2 层)

VLM 中唯一的可训练层。把 CLIP 视觉特征 [B, N, vision_dim]
投影到 LLM embedding space [B, N, llm_dim]。

设计参考: LLaVA v1.5 论文
    "Improved Baselines with Visual Instruction Tuning" (Liu et al., 2024)
    消融实验证明: 2 层 MLP 优于 Q-Former,
    - 参数量少 10x
    - 训练成本低 10x
    - 下游榜单精度普遍更优或持平
"""
import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """两层 MLP, GELU 激活。"""

    def __init__(
        self,
        input_dim: int = 1024,   # CLIP-ViT-L=1024, CLIP-ViT-B=768
        hidden_dim: int = 2048,  # 中间隐层
        # 目标 LLM 的 hidden_size: ScratchVLM 会自动从 llm.config 读并覆盖
        # 参考值: CLIP-L=1024, Qwen3-0.6B=1024
        output_dim: int = 960,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dtype = dtype

        self.linear1 = nn.Linear(input_dim, hidden_dim, dtype=dtype)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, output_dim, dtype=dtype)

        # 初始化: Xavier normal (与 LLaVA 官方保持一致)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """前向: 视觉特征 → LLM embedding 空间。

        Args:
            visual_features: [B, N, input_dim]

        Returns:
            projected: [B, N, output_dim]
        """
        x = self.linear1(visual_features)
        x = self.activation(x)
        x = self.linear2(x)
        return x

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
