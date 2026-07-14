# S2 · 航拍/DJI 领域适配 执行 spec(capstone)

**定位**:ROADMAP P5,叙事价值最高(直连 DJI 主业),优先级最后。目标——把当前 SOTA VLM 适配到**航拍/俯视图像**,把"通用 caption demo"变成"懂无人机场景",并再次演示"冻结主干 + 微型 adapter"的**领域迁移经济学**(只需微调 3.94M projector)。

## 1 · 一个必须诚实面对的领域细节

- **DJI 主业 = 低空、斜视/俯视的无人机航拍**(消费机/行业机/农业/测绘)。
- **带 caption 的低空无人机数据极少**;主流"航拍 captioning"数据集其实是**卫星/高空遥感**(overhead nadir),与低空 drone 有 domain gap。
- 因此以下候选是**最接近的可得代理**,面试时要如实说"这是遥感 overhead 代理,非低空 drone;真实 DJI 数据需内部采集"。

## 2 · 数据集候选

| 数据集 | 规模 | 大小 | 类型 | 适配性 |
|--------|------|------|------|--------|
| **UCM-Captions** | 2100 图 × 5 cap | ~300 MB | UC Merced 土地利用 overhead | 小、快、经典;类别单一 |
| **RSICD** | 10921 图 × 5 cap | ~1.7 GB | 遥感 overhead(标准榜) | 数据多、可对标;下载较大 |
| **Sydney/NWPU-Captions** | 0.6k–31k | 变化 | 遥感 overhead | 备选 |
| 用户自备 DJI 航拍 + 人工/自动 caption | — | — | **真·低空 drone** | 叙事最强,但需采集+标注 |

## 3 · 执行步骤(数据就位后一条龙)

复用现有 pipeline,**从当前 SOTA ckpt 微调**(同 Instruct 微调配方):

```bash
# 1. 数据转 JSONL ({"image","caption"}) + 900/100(或按数据集) image-level split
#    (可复用 data/split_flickr.py 的思路)

# 2. 从 SOTA ckpt 微调 projector(小 lr、短程,冻结主干不变)
python train.py --data data/aerial/train.jsonl --val-data data/aerial/val.jsonl \
                --image-root data/aerial/images \
                --vision openai/clip-vit-large-patch14-336 \
                --llm Qwen/Qwen2.5-0.5B-Instruct \
                --init-projector checkpoints/projector_L14_qwenInstruct_ft_best.pt \
                --steps 400 --batch 4 --grad-accum 4 --lr 2e-4 --warmup 0 \
                --dtype bf16 --val-every 50 \
                --out checkpoints/projector_aerial.pt

# 3. 对比评测:领域 baseline(未适配的通用模型) vs 适配后
python evaluate.py --ckpt checkpoints/projector_L14_qwenInstruct_ft_best.pt \
                   --data data/aerial/val.jsonl --image-root data/aerial/images --max-samples 100  # 领域前
python evaluate.py --ckpt checkpoints/projector_aerial.pt \
                   --data data/aerial/val.jsonl --image-root data/aerial/images --max-samples 100  # 领域后
```

## 4 · 预期与叙事

- **领域 gap 证据**:通用模型(Flickr 训)直接跑航拍图,BLEU 应明显低于领域内 → 量化"domain gap"。
- **适配收益**:仅微调 3.94M projector(冻结 CLIP+LLM),BLEU 回升 → 再次证明"微型 adapter 领域迁移"经济学,且**推理显存不变(仍 1.6GB,仍 Orin 可部署)**。
- **面试话术**:"同一套端侧架构,换个 projector 就适配到航拍域,主体权重零改动——这正是端侧多任务/多域部署的经济性。"(诚实标注:overhead 遥感代理,非低空 drone。)

## 5 · 唯一待定(需 green-light)

**选哪个数据集 + 是否下载**(外部大文件):UCM(小快)/ RSICD(多可对标)/ 用户自备 DJI 数据 / 暂缓。
数据就位后,§3 一条龙约 30-60 分钟出结果。
