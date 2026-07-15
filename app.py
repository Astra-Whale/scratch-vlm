"""
scratch-vlm · Gradio 演示 (面试现场用)

上传图片 → 生成 caption,实时展示端侧指标:推理显存峰值 + latency。

用法:
  python app.py            # 本地 http://127.0.0.1:7860
  python app.py --share    # 生成公网临时链接
"""
import os
import sys
import time
import argparse
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import gradio as gr
from PIL import Image

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
from model.vlm import ScratchVLM, IMAGE_TOKEN

CKPT = os.environ.get("VLM_CKPT", "checkpoints/projector_stage1_qwen3_best.pt")
PROMPT = (f"<|im_start|>user\n{IMAGE_TOKEN}\nDescribe this image.<|im_end|>\n"
          f"<|im_start|>assistant\n")

_STATE = {"model": None}


def load_model():
    """按需加载模型 (bf16), 缓存在 _STATE。"""
    if _STATE["model"] is not None:
        return _STATE["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    m = ScratchVLM(
        vision_model_name=ck.get("vision_name", "openai/clip-vit-large-patch14-336"),
        llm_model_name=ck.get("llm_name", "weights/Qwen3-0.6B"),
        dtype=torch.bfloat16, device=device,
    ).to(device)
    m.projector.load_state_dict(ck["projector_state_dict"])
    m.eval()
    _STATE["model"] = m
    return m


def caption(image: Image.Image, max_new_tokens: int):
    if image is None:
        return "请先上传一张图片。", ""
    m = load_model()
    device = m.llm.device
    pv = m.vision_encoder.image_processor(images=image.convert("RGB"),
                                          return_tensors="pt")["pixel_values"].to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = m.generate(pixel_values=pv, prompt=PROMPT, max_new_tokens=max_new_tokens, temperature=0.0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    text = out["clean"] if isinstance(out, dict) else out
    n = out["num_new_tokens"] if isinstance(out, dict) else max_new_tokens
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        metrics = (f"推理显存峰值: {peak:.0f} MB"
                   f"  |  {dt * 1000:.0f} ms / {n} tok = {n / dt:.1f} tok/s")
    else:
        metrics = f"{dt * 1000:.0f} ms (CPU)"
    return text, metrics


def build_ui():
    with gr.Blocks(title="scratch-vlm · 端侧 VLM demo") as demo:
        gr.Markdown(
            "# scratch-vlm · 散装端侧 VLM\n"
            "CLIP-ViT-L/14@336 (冻结) + MLP projector (4.20M, 唯一可训) + "
            "Qwen3-0.6B (冻结)。上传图片生成描述,观察端侧显存/延迟。"
        )
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="pil", label="输入图片")
                mnt = gr.Slider(8, 60, value=30, step=1, label="max_new_tokens")
                btn = gr.Button("生成描述", variant="primary")
            with gr.Column():
                out_text = gr.Textbox(label="生成的 caption", lines=3)
                out_metrics = gr.Textbox(label="端侧指标", lines=2)
        btn.click(caption, inputs=[img, mnt], outputs=[out_text, out_metrics])
        ex_dir = _ROOT / "data" / "flickr8k" / "images"
        if ex_dir.is_dir():
            exs = [str(p) for p in sorted(ex_dir.glob("*.jpg"))[:6]]
            if exs:
                gr.Examples(examples=exs, inputs=img)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build_ui().launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
