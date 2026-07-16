"""合并 stage-2 LoRA 进 Qwen3-0.6B, 导出可转 GGUF 的完整 HF 模型。

用 ScratchVLM 保证 <image> token 的 resize 与训练一致, 再对 model.llm
(PeftModel) 做 merge_and_unload, 保存合并后的 Qwen3 + tokenizer。

用法:
  python tools/merge_lora.py            # 默认合并 vlm_stage2_mix2 → weights/qwen3_stage2_merged
  python tools/merge_lora.py --lora checkpoints/vlm_stage2_mix/lora_adapter --out weights/qwen3_stage2_mix_merged
"""
import os, sys, argparse
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.vlm import ScratchVLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora", default="checkpoints/vlm_stage2_mix2/lora_adapter",
                   help="LoRA adapter 目录")
    p.add_argument("--out", default="weights/qwen3_stage2_merged",
                   help="合并后 HF 模型输出目录")
    p.add_argument("--vision", default="openai/clip-vit-large-patch14-336")
    p.add_argument("--llm", default="weights/Qwen3-0.6B")
    args = p.parse_args()

    model = ScratchVLM(args.vision, args.llm, dtype=torch.float16, device="cpu")
    from peft import PeftModel
    model.llm = PeftModel.from_pretrained(model.llm, args.lora)
    print(f"[merge] LoRA 已加载自 {args.lora}, merge_and_unload ...")
    merged = model.llm.merge_and_unload()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    model.tokenizer.save_pretrained(str(out))
    print(f"[merge] 合并模型 + tokenizer 已存 {out} "
          f"(vocab_size={merged.config.vocab_size} hidden={merged.config.hidden_size})")


if __name__ == "__main__":
    main()
