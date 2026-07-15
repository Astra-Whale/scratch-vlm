"""
Week 2 D2 · Projector 训练脚本

只训 MLP Projector, 其他全部冻结。用于验证:
    (1) 训练闭环 pipeline 通
    (2) toy 数据上 loss 明显下降 (证明 projector 在学 vision→language 对齐)
    (3) checkpoint 保存/加载正常

用法:
    # 用默认 toy 数据训 30 步
    python train.py

    # 自定义
    python train.py --data data/toy.jsonl --steps 60 --batch 4 --lr 1e-3

期望输出:
    - stdout 打印每步 loss
    - 训练日志写到 logs/train_run_YYYYMMDD_HHMMSS.log
    - projector 权重保存到 checkpoints/projector_stage1_qwen3.pt
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

# Windows 控制台 UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from torch.utils.data import DataLoader

# 允许 python train.py 直接跑, 添加项目根到 sys.path
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from model.vlm import ScratchVLM, IMAGE_TOKEN
from data.dataset import MVPCaptionDataset, collate_fn


# ==========================================================
# CLI 参数
# ==========================================================
def parse_args():
    p = argparse.ArgumentParser(description="train MLP projector for scratch-vlm")
    # 数据
    p.add_argument("--data", type=str, default="data/flickr8k/train.jsonl",
                   help="JSONL 数据集路径")
    p.add_argument("--image-root", type=str, default="data/flickr8k/images",
                   help="图像根目录 (相对路径解析基准)")
    p.add_argument("--question", type=str, default="Describe this image.",
                   help="固定问句 (放在 <image> 之后)")
    # 模型
    p.add_argument("--vision", type=str, default="openai/clip-vit-large-patch14-336",
                   help="CLIP 模型 (本地路径或 HF repo id)")
    p.add_argument("--llm", type=str, default="weights/Qwen3-0.6B",
                   help="LLM (HF repo id 或本地路径)")
    p.add_argument("--init-projector", type=str, default=None,
                   help="从已有 checkpoint 载入 projector 权重作为初始化 (微调/续训); "
                        "维度必须与当前 vision/llm 匹配。默认 None = Xavier 随机初始化")
    # 训练
    p.add_argument("--steps", type=int, default=30, help="总训练步数")
    p.add_argument("--batch", type=int, default=4, help="micro-batch size (单次前向)")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="梯度累积步数; 有效 batch = batch * grad_accum。"
                        "用于在显存受限时(如 CLIP-L@336 576 tokens)凑大有效 batch")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p.add_argument("--warmup", type=int, default=3, help="warmup 步数 (按 optimizer step 计)")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"],
                   help="训练精度 (bf16 更稳, fp16 更快但可能 underflow)")
    # Val 相关 (可选)
    p.add_argument("--val-data", type=str, default=None,
                   help="Val JSONL 路径 (可选, 提供则周期性计算 val loss)")
    p.add_argument("--val-every", type=int, default=25,
                   help="每多少步跑一次 val")
    p.add_argument("--val-batches", type=int, default=4,
                   help="每次 val 采样多少 batch (快速估算, 不遍历全部 val)")
    # 输出
    p.add_argument("--out", type=str, default="checkpoints/projector_stage1_qwen3.pt",
                   help="保存 projector 权重的路径")
    p.add_argument("--log-dir", type=str, default="logs", help="日志目录")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ==========================================================
# 学习率调度: linear warmup + cosine decay
# ==========================================================
def get_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    # cosine from 1.0 → 0.1
    import math
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


# ==========================================================
# main
# ==========================================================
def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    print(f"[env] device={device}, dtype={args.dtype}")

    # ---- 加载模型 ----
    print(f"[model] vision={args.vision}")
    print(f"[model] llm={args.llm}")
    t0 = time.time()
    model = ScratchVLM(
        vision_model_name=args.vision,
        llm_model_name=args.llm,
        dtype=dtype,
        device=device,
    ).to(device)
    print(f"[model] 加载耗时 {time.time() - t0:.1f}s")

    # 训练时关闭 KV-cache (只在 generate 时需要; 训练前向留着会有 warning 且略费显存)
    if hasattr(model.llm, "config"):
        model.llm.config.use_cache = False

    # 冻结一切, 仅解冻 projector
    for p in model.parameters():
        p.requires_grad = False
    for p in model.projector.parameters():
        p.requires_grad = True

    # 可选: 从已有 checkpoint 载入 projector 作为初始化 (微调/续训)
    if args.init_projector:
        init_ck = torch.load(args.init_projector, map_location="cpu", weights_only=False)
        model.projector.load_state_dict(init_ck["projector_state_dict"])
        print(f"[model] projector 初始化自 {args.init_projector} "
              f"(原 trained_steps={init_ck.get('trained_steps')}, "
              f"val_loss={init_ck.get('val_loss')})")

    trainable = sum(p.numel() for p in model.projector.parameters())
    print(f"[model] 可训参数 (projector): {trainable / 1e6:.2f}M")

    # ---- 数据 ----
    ds = MVPCaptionDataset(
        jsonl_path=args.data,
        image_root=args.image_root,
        image_processor=model.vision_encoder.image_processor,
        tokenizer=model.tokenizer,
        image_token=IMAGE_TOKEN,
        question=args.question,
    )
    print(f"[data] {len(ds)} 条样本 from {args.data}")

    pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_id),
    )

    # Val loader (可选)
    val_loader = None
    if args.val_data:
        val_ds = MVPCaptionDataset(
            jsonl_path=args.val_data,
            image_root=args.image_root,
            image_processor=model.vision_encoder.image_processor,
            tokenizer=model.tokenizer,
            image_token=IMAGE_TOKEN,
            question=args.question,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch,
            shuffle=True,  # 每次 val 随机抽 val_batches 个, 避免每次同样
            num_workers=0,
            collate_fn=lambda b: collate_fn(b, pad_token_id=pad_id),
        )
        print(f"[data] val: {len(val_ds)} samples from {args.val_data}")

    # Val loss 计算函数
    def compute_val_loss():
        assert val_loader is not None
        model.projector.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for i, b in enumerate(val_loader):
                if i >= args.val_batches:
                    break
                b = {k: v.to(device) for k, v in b.items()}
                out = model(**b)
                total += out.loss.item()
                n += 1
        model.projector.train()
        return total / max(n, 1)

    # ---- 优化器 ----
    optim = torch.optim.AdamW(model.projector.parameters(), lr=args.lr, weight_decay=0.0)

    # ---- 日志文件 ----
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_run_{ts}.log"
    log_lines = []

    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    log(f"[env] device={device}, dtype={args.dtype}")
    log(f"[model] projector trainable: {trainable / 1e6:.2f}M")
    log(f"[data]  {len(ds)} samples, batch={args.batch}, steps={args.steps}, lr={args.lr}")

    # ---- 训练 loop ----
    # projector 训练模式; vision/llm 保持 eval (init 时已 .eval())
    model.projector.train()

    step = 0
    losses = []
    val_history = []  # list of (step, val_loss)
    best_val = float("inf")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t_start = time.time()
    ga = max(1, args.grad_accum)
    if ga > 1:
        log(f"[train] grad_accum={ga}, 有效 batch={args.batch * ga}")
    optim.zero_grad()
    data_iter = iter(loader)
    while step < args.steps:
        # 累积 ga 个 micro-batch 的梯度 = 1 个 optimizer step
        accum_loss = 0.0
        for _ in range(ga):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / ga  # 归一化, 使累积梯度等价于大 batch 平均
            loss.backward()
            accum_loss += out.loss.item()

        # 手动 LR schedule (按 optimizer step)
        lr = get_lr(step, args.steps, args.warmup, args.lr)
        for g in optim.param_groups:
            g["lr"] = lr

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.projector.parameters(), max_norm=1.0
        )
        optim.step()
        optim.zero_grad()

        mean_loss = accum_loss / ga
        losses.append(mean_loss)
        log(f"[step {step:3d}] loss={mean_loss:.4f}  lr={lr:.2e}  |grad|={grad_norm.item():.3f}")

        # Val loss 周期评估
        if val_loader is not None and (step + 1) % args.val_every == 0:
            vl = compute_val_loss()
            val_history.append((step, vl))
            marker = ""
            if vl < best_val:
                best_val = vl
                marker = " ← best"
                # 保存 best-val checkpoint
                best_path = Path(args.out).with_stem(Path(args.out).stem + "_best")
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "projector_state_dict": model.projector.state_dict(),
                    "vision_hidden_size": model.projector.input_dim,
                    "llm_hidden_size": model.projector.output_dim,
                    "vision_name": args.vision,
                    "llm_name": args.llm,
                    "trained_steps": step + 1,
                    "val_loss": vl,
                }, best_path)
            log(f"           └ val_loss={vl:.4f}{marker}")

        step += 1

    t_end = time.time()

    # ---- 保存 checkpoint ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "projector_state_dict": model.projector.state_dict(),
            "vision_hidden_size": model.projector.input_dim,
            "llm_hidden_size": model.projector.output_dim,
            "vision_name": args.vision,
            "llm_name": args.llm,
            "trained_steps": step,
            "final_loss": losses[-1] if losses else None,
        },
        out_path,
    )
    log(f"[save] projector → {out_path}")

    # ---- 汇总 ----
    log(f"[done] {step} steps in {t_end - t_start:.1f}s "
        f"({(t_end - t_start) / max(1, step) * 1000:.0f} ms/step)")
    if losses:
        first_loss = sum(losses[:3]) / len(losses[:3])
        last_loss = sum(losses[-3:]) / len(losses[-3:])
        drop_pct = (first_loss - last_loss) / first_loss * 100
        log(f"[loss] 首 3 步均值 {first_loss:.4f} → 末 3 步均值 {last_loss:.4f}  "
            f"(下降 {drop_pct:.1f}%)")

    if val_history:
        log("[val] history:")
        for s, v in val_history:
            log(f"       step {s}: {v:.4f}")
        log(f"[val] best={best_val:.4f} (checkpoint 存 {Path(args.out).with_stem(Path(args.out).stem + '_best')})")

    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        log(f"[vram] 峰值 {peak_mb:.0f} MB")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[log] 保存到 {log_path}")


if __name__ == "__main__":
    main()
