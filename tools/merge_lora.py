"""Phase A: 合并 stage-2 LoRA 进 Qwen3-0.6B, 导出可转 GGUF 的完整 HF 模型。

用 ScratchVLM 保证 <image> token 的 resize 与训练一致, 再对 model.llm
(PeftModel) 做 merge_and_unload, 保存合并后的 Qwen3 + tokenizer。
"""
import os, sys
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.vlm import ScratchVLM

OUT = Path("weights/qwen3_stage2_merged")
LORA = "checkpoints/vlm_stage2_mix2/lora_adapter"

model = ScratchVLM("openai/clip-vit-large-patch14-336", "weights/Qwen3-0.6B",
                   dtype=torch.float16, device="cpu")
from peft import PeftModel
model.llm = PeftModel.from_pretrained(model.llm, LORA)
print("[merge] LoRA 已加载, merge_and_unload ...")
merged = model.llm.merge_and_unload()
OUT.mkdir(parents=True, exist_ok=True)
merged.save_pretrained(str(OUT), safe_serialization=True)
model.tokenizer.save_pretrained(str(OUT))
print(f"[merge] 合并模型 + tokenizer 已存 {OUT}")
print(f"[merge] vocab_size={merged.config.vocab_size} hidden={merged.config.hidden_size}")
