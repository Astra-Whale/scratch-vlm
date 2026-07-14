"""
视觉前端 ONNX 导出 (端侧部署路径第一步)

导出 "pixels → visual tokens" 的完整前端:
  CLIP-ViT-L/14@336 vision tower (select_layer=-2, patch) + MLP projector
  输入 [B,3,336,336] → 输出 [B,576,896] (LLM embedding 空间的视觉 token)

这是 PyTorch → ONNX → (Jetson 上) TensorRT engine 部署链路的第一步。
ONNX 本身硬件无关、可移植;TensorRT engine 构建是目标平台特定的(诚实说明)。
用 fp32 导出以保证 onnxruntime / TensorRT 兼容性(权重精度由目标端再定)。

用法:
  python benchmark/export_onnx.py --ckpt checkpoints/projector_L14_qwenInstruct_ft_best.pt
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
from model.vlm import ScratchVLM


class VisualFrontend(nn.Module):
    """pixels → projected visual tokens (LLM embedding 空间)。"""
    def __init__(self, model: ScratchVLM):
        super().__init__()
        self.vision_encoder = model.vision_encoder
        self.projector = model.projector

    def forward(self, pixel_values):
        feat = self.vision_encoder(pixel_values)   # [B, 576, 1024]
        return self.projector(feat)                 # [B, 576, 896]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/projector_L14_qwenInstruct_ft_best.pt")
    p.add_argument("--out", type=str, default="onnx/visual_frontend.onnx")
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def main():
    args = parse_args()
    # fp32 + CPU 导出 (便携; 视觉前端不大, CPU 足够)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = ScratchVLM(
        vision_model_name=ck.get("vision_name", "openai/clip-vit-large-patch14-336"),
        llm_model_name=ck.get("llm_name", "Qwen/Qwen2.5-0.5B-Instruct"),
        dtype=torch.float32, device="cpu",
    )
    model.projector.load_state_dict(ck["projector_state_dict"])
    model.eval()

    frontend = VisualFrontend(model).eval()
    dummy = torch.randn(1, 3, 336, 336, dtype=torch.float32)

    with torch.no_grad():
        ref = frontend(dummy)
    print(f"[torch] 输出 shape {tuple(ref.shape)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        frontend, (dummy,), args.out,
        input_names=["pixel_values"], output_names=["visual_tokens"],
        dynamic_axes={"pixel_values": {0: "batch"}, "visual_tokens": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
    )
    # 新版 dynamo 导出器把权重存为外部 .data 文件, 统计时一并计入
    size_mb = Path(args.out).stat().st_size / 1024 ** 2
    data_file = Path(str(args.out) + ".data")
    if data_file.exists():
        size_mb += data_file.stat().st_size / 1024 ** 2
    print(f"[onnx] 导出 → {args.out} (含外部权重共 {size_mb:.0f} MB fp32, opset {args.opset})")

    # 结构校验
    import onnx
    onnx.checker.check_model(onnx.load(args.out))
    print("[onnx] checker.check_model 通过")

    # onnxruntime 数值一致性校验
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"pixel_values": dummy.numpy()})[0]
    max_abs = float(torch.max(torch.abs(ref - torch.from_numpy(ort_out))))
    # 深层 ViT (24 层) fp32 累积舍入 + ONNX 图优化, ~1e-2 量级属正常
    print(f"[verify] torch vs onnxruntime 最大绝对误差: {max_abs:.2e} "
          f"({'一致 ✓ (fp32 累积级)' if max_abs < 1e-2 else '偏差偏大 ⚠'})")


if __name__ == "__main__":
    main()
