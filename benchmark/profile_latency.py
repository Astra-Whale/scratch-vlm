"""
推理 latency 分解 profiler (端侧视角)

拆解单图 caption 生成的耗时:
  1. 视觉编码 (CLIP-ViT encode + MLP projector): 一次性开销, 与生成长度无关
  2. LLM prefill (首 token, 含 576 visual tokens + prompt 的一次前向)
  3. LLM per-token decode (自回归, batch=1 下 memory-bandwidth-bound)

方法: 对 generate(max_new_tokens=N) 在多个 N 上计时,
  decode_per_token = (t(N2) - t(N1)) / (N2 - N1)
  prefill ≈ t(1) - decode_per_token
输出 stdout + logs/latency_profile.json。

用法:
  python benchmark/profile_latency.py --ckpt checkpoints/projector_L14_qwenInstruct_ft_best.pt
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
from PIL import Image

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
from model.vlm import ScratchVLM, IMAGE_TOKEN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/projector_L14_qwenInstruct_ft_best.pt")
    p.add_argument("--image", type=str, default="data/flickr_1k/images/6317293855.jpg")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--iters", type=int, default=8, help="每个测点重复次数取均值")
    p.add_argument("--out", type=str, default="logs/latency_profile.json")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = ScratchVLM(
        vision_model_name=ck.get("vision_name", "openai/clip-vit-large-patch14-336"),
        llm_model_name=ck.get("llm_name", "Qwen/Qwen2.5-0.5B-Instruct"),
        dtype=dtype, device=device,
    ).to(device)
    model.eval()
    model.projector.load_state_dict(ck["projector_state_dict"])

    image = Image.open(args.image).convert("RGB")
    pv = model.vision_encoder.image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    prompt = (f"<|im_start|>user\n{IMAGE_TOKEN}\nDescribe this image.<|im_end|>\n"
              f"<|im_start|>assistant\n")

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    # warmup
    for _ in range(3):
        model.generate(pixel_values=pv, prompt=prompt, max_new_tokens=10, temperature=0.0)
    sync()

    K = args.iters

    # 1) 视觉编码 (CLIP encode + projector)
    t = 0.0
    for _ in range(K):
        sync(); s = time.time()
        _ = model.encode_images(pv)
        sync(); t += time.time() - s
    t_vis = t / K * 1000

    # 预备 inputs_embeds (复刻 vlm.generate 的拼接), 用于强制固定长度解码
    inputs = model.tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        visual = model.encode_images(pv)
        text_embeds = model.llm.get_input_embeddings()(input_ids)
    p = (input_ids[0] == model.image_token_id).nonzero(as_tuple=True)[0][0].item()
    embeds = torch.cat([text_embeds[0, :p], visual[0], text_embeds[0, p + 1:]], dim=0).unsqueeze(0)
    attn = torch.ones(embeds.shape[:2], dtype=torch.long, device=device)

    # 2) 强制精确生成 N 个 token (禁 eos + min_new_tokens=max_new_tokens), 隔离纯 decode 成本
    @torch.no_grad()
    def gen_time_fixed(n):
        tt = 0.0
        for _ in range(K):
            sync(); s = time.time()
            model.llm.generate(
                inputs_embeds=embeds, attention_mask=attn,
                min_new_tokens=n, max_new_tokens=n,
                do_sample=False, eos_token_id=None,
                pad_token_id=model.tokenizer.eos_token_id,
            )
            sync(); tt += time.time() - s
        return tt / K * 1000

    g1, g16, g64 = gen_time_fixed(1), gen_time_fixed(16), gen_time_fixed(64)
    decode_per_tok = (g64 - g16) / 48.0
    prefill = g1 - decode_per_tok  # g1 含 prefill(576 visual + prompt 前向) + 1 token decode
    g32 = gen_time_fixed(32)

    res = {
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "dtype": args.dtype,
        "vision_encode_ms": round(t_vis, 1),
        "generate_1tok_ms": round(g1, 1),
        "generate_16tok_ms": round(g16, 1),
        "generate_32tok_ms": round(g32, 1),
        "prefill_ms_incl_576visual_prompt": round(prefill, 1),
        "decode_per_token_ms": round(decode_per_tok, 2),
        "decode_tok_per_s": round(1000.0 / decode_per_tok, 1),
        "vision_share_of_32tok_pct": round(t_vis / g32 * 100, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"Latency 分解 · {res['device']} · {args.dtype} · batch=1")
    print("=" * 60)
    for k, v in res.items():
        print(f"  {k:38}= {v}")
    print()
    print(f"  → 视觉编码是一次性开销 (占 32-token 生成的 {res['vision_share_of_32tok_pct']}%);")
    print(f"    LLM decode {res['decode_tok_per_s']} tok/s (batch=1 memory-bandwidth-bound)")


if __name__ == "__main__":
    main()
