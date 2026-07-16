"""
前向验证测试

目标:
1. 加载 CLIP + Projector + Qwen 拼装 VLM 无报错
2. 参数量核对: 仅 projector 可训
3. 用假数据跑一次前向 pipeline (含 <image> token 替换)
4. shape 对齐、loss 有值
5. VRAM 占用符合预期 (<6GB)

首次运行需下载 CLIP-L/14@336 (~1.7GB) + Qwen3-0.6B (~1.2GB)。
"""
import os
import sys
import time

import torch

# 让 model 模块能被 import (tests/ 目录下)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from model.vlm import ScratchVLM, IMAGE_TOKEN  # noqa: E402


def _banner(msg: str, char: str = "=", width: int = 60):
    print(char * width)
    print(msg)
    print(char * width)


def test_shapes_and_forward():
    _banner("scratch-vlm 前向验证测试")

    # ============ [env] 环境自检 ============
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"\n[env] device={device}, dtype={dtype}")
    if device == "cuda":
        print(f"      GPU: {torch.cuda.get_device_name(0)}")
        print(f"      VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.reset_peak_memory_stats()

    # ============ [1] 加载 VLM ============
    # ============ 模型选型 ============
    # 默认用当前架构: CLIP-ViT-L/14@336 + Qwen3-0.6B。
    # 通过环境变量 VLM_VISION / VLM_LLM 可覆盖默认。
    _local_qwen3 = os.path.join(_ROOT, "weights", "Qwen3-0.6B")
    default_llm = _local_qwen3 if os.path.isdir(_local_qwen3) else "Qwen/Qwen3-0.6B"

    vision_name = os.environ.get("VLM_VISION", "openai/clip-vit-large-patch14-336")
    llm_name = os.environ.get("VLM_LLM", default_llm)

    print(f"\n[1] 加载 VLM (vision={vision_name}, llm={llm_name})...")
    t0 = time.time()
    model = ScratchVLM(
        vision_model_name=vision_name,
        llm_model_name=llm_name,
        dtype=dtype,
        device=device,
    ).to(device)
    print(f"    ✓ 加载完成 (耗时 {time.time() - t0:.1f}s)")

    # ============ [2] 参数量统计 ============
    print("\n[2] 参数量统计")
    total = model.num_total_parameters()
    trainable = model.num_trainable_parameters()
    projector_params = sum(p.numel() for p in model.projector.parameters())

    print(f"    总参数量:      {total / 1e6:.1f}M")
    print(f"    可训参数量:    {trainable / 1e6:.2f}M")
    print(f"    Projector 参数: {projector_params / 1e6:.2f}M")

    assert trainable == projector_params, (
        f"可训参数应严格等于 projector 参数, "
        f"but trainable={trainable}, projector={projector_params}"
    )
    print("    ✓ 仅 projector 可训, CLIP + LLM 已冻结")

    # ============ [3] 构造假输入 ============
    print("\n[3] 构造假输入 (batch=2, 336×336 图, prompt=IMAGE_TOKEN + 中文描述指令)")
    fake_img = torch.randn(2, 3, 336, 336).to(device)

    prompt_text = f"{IMAGE_TOKEN}\n描述这张图片。"
    tokenized = model.tokenizer(
        [prompt_text, prompt_text],
        return_tensors="pt",
        padding=True,
    ).to(device)
    input_ids = tokenized["input_ids"]
    attn_mask = tokenized["attention_mask"]
    print(f"    input_ids shape: {tuple(input_ids.shape)}")
    print(f"    image_token_id: {model.image_token_id}")
    print(f"    每条含 <image> token 数: {(input_ids == model.image_token_id).sum(dim=1).tolist()}")

    # ============ [4] 前向传播 ============
    print("\n[4] 前向传播 (labels=input_ids.clone() 触发 loss 计算)...")
    t0 = time.time()
    out = model(
        pixel_values=fake_img,
        input_ids=input_ids,
        attention_mask=attn_mask,
        labels=input_ids.clone(),
    )
    fwd_time = time.time() - t0
    print(f"    ✓ forward 通过 (耗时 {fwd_time * 1000:.1f}ms)")
    print(f"    logits shape: {tuple(out.logits.shape)}")
    print(f"    loss: {out.loss.item():.4f}")

    # ============ [5] VRAM 占用 ============
    print("\n[5] VRAM 峰值占用")
    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        alloc_mb = torch.cuda.memory_allocated() / 1024**2
        print(f"    peak: {peak_mb:.0f} MB")
        print(f"    now:  {alloc_mb:.0f} MB")
        if peak_mb < 6000:
            print("    ✓ VRAM 占用在预期范围 (<6 GB)")
        else:
            print(f"    ⚠ VRAM 占用偏高: {peak_mb:.0f} MB (预期 <6 GB)")
    else:
        print("    (CPU 模式, 不测 VRAM)")

    # ============ 结果 ============
    print()
    _banner("✓ 前向 pipeline 通过", char="=", width=60)


if __name__ == "__main__":
    test_shapes_and_forward()
