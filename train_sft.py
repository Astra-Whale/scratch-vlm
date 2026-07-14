"""
stage-2 · 视觉指令微调 (LoRA + projector 联合 SFT) · 方法对齐 LLaVA v1.5

- CLIP 冻结; Qwen base 冻结, **注意力层 q_proj/v_proj 加 LoRA**; projector 从 stage-1 初始化后继续训。
- 数据: LLaVA-Instruct 对话格式 (data/sft_dataset.py), 只对 gpt turn 算 loss。
- 保存: projector 权重 + LoRA adapter。

用法:
  python train_sft.py --data data/llava_instruct/detail_23k.json \
      --image-root data/coco/train2014 \
      --vision openai/clip-vit-large-patch14-336 --llm Qwen/Qwen3-0.6B \
      --init-projector checkpoints/projector_stage1_qwen3.pt \
      --steps 800 --batch 4 --grad-accum 4 --lr 2e-4 --lora-rank 16 \
      --out checkpoints/vlm_stage2_lora
"""
import os, sys, math, time, argparse
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
from model.vlm import ScratchVLM, IMAGE_TOKEN
from data.sft_dataset import LlavaInstructDataset
from data.dataset import collate_fn


def get_lr(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--vision", default="openai/clip-vit-large-patch14-336")
    p.add_argument("--llm", default="Qwen/Qwen3-0.6B")
    p.add_argument("--init-projector", default=None, help="stage-1 projector ckpt")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--out", default="checkpoints/vlm_stage2_lora")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print(f"[model] vision={args.vision} llm={args.llm}")
    model = ScratchVLM(args.vision, args.llm, dtype=dtype, device=device).to(device)
    if hasattr(model.llm, "config"):
        model.llm.config.use_cache = False
    if args.init_projector:
        ck = torch.load(args.init_projector, map_location="cpu", weights_only=False)
        model.projector.load_state_dict(ck["projector_state_dict"])
        print(f"[model] projector 初始化自 {args.init_projector}")

    # ---- LoRA on Qwen attention q/v ----
    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM",
    )
    model.llm = get_peft_model(model.llm, lora_cfg)
    # projector 保持可训 (LoRA 已自动只解冻 adapter, CLIP 冻结)
    for p in model.projector.parameters():
        p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_lora = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora" in n.lower())
    n_proj = sum(p.numel() for p in model.projector.parameters())
    print(f"[model] 可训 {n_train/1e6:.2f}M (LoRA {n_lora/1e6:.2f}M + projector {n_proj/1e6:.2f}M)")

    ds = LlavaInstructDataset(args.data, args.image_root,
                              model.vision_encoder.image_processor, model.tokenizer,
                              image_token=IMAGE_TOKEN)
    print(f"[data] {len(ds)} 条本地有图样本 from {args.data}")
    pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2,
                        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_id))

    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    log_dir = Path(args.log_dir); log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logf = log_dir / f"sft_run_{ts}.log"; lines = []
    def log(m): print(m); lines.append(m)
    log(f"[sft] steps={args.steps} batch={args.batch} accum={args.grad_accum} lr={args.lr} lora_r={args.lora_rank}")

    model.train()
    if device == "cuda": torch.cuda.reset_peak_memory_stats()
    ga = max(1, args.grad_accum); step = 0; losses = []
    optim.zero_grad(); it = iter(loader); t0 = time.time()
    while step < args.steps:
        acc = 0.0
        for _ in range(ga):
            try: batch = next(it)
            except StopIteration: it = iter(loader); batch = next(it)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            (out.loss / ga).backward(); acc += out.loss.item()
        lr = get_lr(step, args.steps, args.warmup, args.lr)
        for g in optim.param_groups: g["lr"] = lr
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step(); optim.zero_grad()
        losses.append(acc / ga)
        if step % 10 == 0 or step == args.steps - 1:
            log(f"[step {step:4d}] loss={acc/ga:.4f} lr={lr:.2e} |g|={gn.item():.2f}")
        step += 1

    dt = time.time() - t0
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model.llm.save_pretrained(str(out / "lora_adapter"))  # peft adapter
    torch.save({
        "projector_state_dict": model.projector.state_dict(),
        "vision_name": args.vision, "llm_name": args.llm,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "trained_steps": step,
        "final_loss": sum(losses[-3:]) / len(losses[-3:]) if losses else None,
    }, out / "projector.pt")
    fl = sum(losses[:3]) / 3; ll = sum(losses[-3:]) / 3
    log(f"[done] {step} steps {dt:.0f}s ({dt/max(1,step)*1000:.0f}ms/step); loss {fl:.3f}→{ll:.3f}")
    if device == "cuda":
        log(f"[vram] 峰值 {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
    log(f"[save] projector+lora → {out}")
    logf.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
