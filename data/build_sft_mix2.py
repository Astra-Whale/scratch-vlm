"""构建 stage-2 SFT 混合数据集 sft_mix2.json(旗舰 POPE 78.59 的训练数据)。

组成(均过滤到本地已有的 COCO train2014 图):
  - LLaVA-Instruct detail_23k     : 图像详细描述
  - LLaVA-Instruct conversation_58k: 多轮视觉对话
  - VQAv2 平衡 yes/no             : 从 VQAv2 抽 answer_type=='yes/no', 下采样成 50/50

平衡 yes/no 是把 POPE 从 all-yes(F1 66.7 虚高)拉到均衡(avg F1 78.59)的关键。

源文件(gitignored, 需先下载, 见 docs/data_sourcing.md):
  - data/llava_instruct/detail_23k.json
  - data/llava_instruct/conversation_58k.json      (liuhaotian/LLaVA-Instruct-150K)
  - data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json
  - data/vqav2/v2_mscoco_train2014_annotations.json (visualqa.org)
  - data/coco/train2014/*.jpg                        (决定过滤范围)

用法:
  python data/build_sft_mix2.py            # 默认路径, 输出 data/llava_instruct/sft_mix2.json
"""
import json
import random
import argparse
from pathlib import Path


def build_vqa_balanced(questions_path, annotations_path, local_images, seed=42):
    """从 VQAv2 抽 yes/no, 过滤到本地图, 下采样成 50/50 平衡, 返回 LLaVA 格式样本。"""
    q = {x["question_id"]: x["question"]
         for x in json.load(open(questions_path, encoding="utf-8"))["questions"]}
    ann = json.load(open(annotations_path, encoding="utf-8"))["annotations"]
    yes_s, no_s = [], []
    for a in ann:
        if a["answer_type"] != "yes/no":
            continue
        fn = f"{a['image_id']:012d}.jpg"
        if fn not in local_images:
            continue
        ans = a["multiple_choice_answer"].strip().lower()
        if ans not in ("yes", "no"):
            continue
        samp = {"id": f"vqa_{a['question_id']}", "image": fn,
                "conversations": [
                    {"from": "human", "value": "<image>\n" + q[a["question_id"]]},
                    {"from": "gpt", "value": ans.capitalize()}]}
        (yes_s if ans == "yes" else no_s).append(samp)
    rng = random.Random(seed)
    rng.shuffle(yes_s); rng.shuffle(no_s)
    k = min(len(yes_s), len(no_s))
    bal = yes_s[:k] + no_s[:k]
    rng.shuffle(bal)
    print(f"[vqa] 本地图 yes={len(yes_s)} no={len(no_s)} -> 平衡各取 {k}, 共 {len(bal)}")
    return bal


def local_only(json_path, local_images):
    data = json.load(open(json_path, encoding="utf-8"))
    return [s for s in data if Path(s["image"]).name in local_images]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-root", default="data/coco/train2014")
    p.add_argument("--detail", default="data/llava_instruct/detail_23k.json")
    p.add_argument("--conversation", default="data/llava_instruct/conversation_58k.json")
    p.add_argument("--vqa-questions", default="data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json")
    p.add_argument("--vqa-annotations", default="data/vqav2/v2_mscoco_train2014_annotations.json")
    p.add_argument("--out", default="data/llava_instruct/sft_mix2.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    local = {p.name for p in Path(args.image_root).glob("*.jpg")}
    print(f"[data] 本地 train2014 图: {len(local)}")

    detail = local_only(args.detail, local)
    conv = local_only(args.conversation, local)
    vqa = build_vqa_balanced(args.vqa_questions, args.vqa_annotations, local, args.seed)
    print(f"[data] detail {len(detail)} + conversation {len(conv)} + 平衡VQA {len(vqa)}")

    mix = detail + conv + vqa
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(mix, open(args.out, "w"), ensure_ascii=False)
    print(f"[out] {len(mix)} 条 -> {args.out}")


if __name__ == "__main__":
    main()
