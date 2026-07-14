"""
Week 2 D2 · 推理脚本

功能:
    加载 projector checkpoint + 一张图 → 生成描述

用法:
    # 用 toy 训完的 projector, 对第 0 张 toy 图跑一次
    python inference.py

    # 指定图和 prompt
    python inference.py --image data/toy_images/000_red_circle.png \
        --prompt "<image>\nDescribe this image."

    # 用不同 checkpoint
    python inference.py --ckpt checkpoints/projector_toy.pt

    # 对比: 未训练的 projector (随机权重)
    python inference.py --no-load

产出:
    stdout 打印生成结果; 若指定 --out 则同时写文件。
"""
import os
import sys
import argparse
from pathlib import Path

# Windows 控制台 UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from PIL import Image

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from model.vlm import ScratchVLM, IMAGE_TOKEN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/projector_toy.pt",
                   help="projector checkpoint (含 state_dict + 模型元数据)")
    p.add_argument("--image", type=str, default="data/toy_images/000_orange_square.png",
                   help="待推理图片 (默认取 toy_images 里的第 000 张; 也可指向 flickr_1k/images/*.jpg)")
    # ChatML 格式 prompt (与 dataset.py 训练时一致, Instruct 模型必须这样)
    _default_prompt = (
        f"<|im_start|>user\n{IMAGE_TOKEN}\nDescribe this image.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    p.add_argument("--prompt", type=str, default=_default_prompt,
                   help="prompt (必须包含 <image> token, 建议 ChatML 格式)")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--no-load", action="store_true",
                   help="不加载 projector 权重 (用于对比随机初始化 baseline)")
    p.add_argument("--vision", type=str, default=None,
                   help="覆盖 vision 模型 (默认从 ckpt 读)")
    p.add_argument("--llm", type=str, default=None,
                   help="覆盖 llm (默认从 ckpt 读)")
    return p.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # ---- 从 ckpt 读模型选型 (除非 CLI 覆盖) ----
    ckpt_path = Path(args.ckpt)
    if ckpt_path.exists() and not args.no_load:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        vision_name = args.vision or ckpt.get("vision_name")
        llm_name = args.llm or ckpt.get("llm_name")
        print(f"[ckpt] 从 {ckpt_path} 读取: trained_steps={ckpt.get('trained_steps')}, "
              f"final_loss={ckpt.get('final_loss')}")
    else:
        # 无 ckpt / no-load: 用默认小模型
        default_vision = (_ROOT / "models" / "models" /
                          "openai-mirror--clip-vit-base-patch32" /
                          "snapshots" / "master").as_posix()
        vision_name = args.vision or default_vision
        llm_name = args.llm or "HuggingFaceTB/SmolLM2-360M-Instruct"
        ckpt = None
        print("[ckpt] 未加载 projector 权重 (随机初始化)")

    # ---- 本地路径 fallback: ckpt 存的 vision 路径若在本机不存在, 回退到本地默认 CLIP ----
    # (��平台迁移场景: ckpt 里可能是旧机器的绝对路径, 如 Windows 的 D:/...)
    # 只针对绝对路径 / Windows 盘符路径; HF repo_id (含 '/') 不碰。
    _is_local_path = vision_name and (
        vision_name[0] in "/\\"
        or "\\" in vision_name
        or (len(vision_name) > 1 and vision_name[1] == ":")
    )
    if _is_local_path and not Path(vision_name).exists():
        _local_default = (_ROOT / "models" / "models" /
                          "openai-mirror--clip-vit-base-patch32" /
                          "snapshots" / "master")
        if _local_default.is_dir():
            print(f"[warn] ckpt 记录的 vision 路径不存在 ({vision_name}), "
                  f"回退到本地默认 CLIP: {_local_default.as_posix()}")
            vision_name = _local_default.as_posix()

    # ---- 加载模型 ----
    print(f"[model] vision={vision_name}")
    print(f"[model] llm={llm_name}")
    model = ScratchVLM(
        vision_model_name=vision_name,
        llm_model_name=llm_name,
        dtype=dtype,
        device=device,
    ).to(device)
    model.eval()

    # ---- 加载 projector 权重 ----
    if ckpt is not None:
        model.projector.load_state_dict(ckpt["projector_state_dict"])
        print(f"[ckpt] projector 权重已加载")

    # ---- 图像 ----
    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"找不到图像: {img_path}")
    image = Image.open(img_path).convert("RGB")
    pixel_values = model.vision_encoder.image_processor(
        images=image, return_tensors="pt"
    )["pixel_values"].to(device)

    # ---- 生成 ----
    print(f"[gen] image={img_path.name}")
    print(f"[gen] prompt={args.prompt!r}")
    print(f"[gen] max_new_tokens={args.max_new_tokens}, temperature={args.temperature}")
    print("-" * 50)
    result = model.generate(
        pixel_values=pixel_values,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(f"new_tokens_generated: {result['num_new_tokens']}")
    print(f"raw (含特殊 token):   {result['raw']!r}")
    print(f"clean (去特殊 token):  {result['clean']!r}")
    print(f"output_ids: {result['output_ids'][:20]}{' ...' if len(result['output_ids']) > 20 else ''}")
    print("-" * 50)


if __name__ == "__main__":
    main()
