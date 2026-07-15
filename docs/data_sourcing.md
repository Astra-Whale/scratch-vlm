# 数据源溯源 + 下载可行性实测

> 目标:为「两阶段训练(阶段1 projector 对齐 / 阶段2 LoRA 视觉指令微调)+ POPE 幻觉评测」补数据。
> 原则:**数据集身份可替代,但用法/格式必须严格对齐 LLaVA v1.5 方法**。本文所有源均在本机 `conda env dl`(python3.11)实测过可达性与下载吞吐,推荐项优先「吞吐已验证」的源。
>
> 实测日期:2026-07-14。实测机:项目 Ubuntu 主机,中文 locale。
> 基线吞吐:`huggingface.co` 直连 ≈ **145–175 KB/s**(稳定,无需梯子);`hf-mirror.com` 镜像实测 ≈ 160 KB/s,**与直连无明显优势**,故一律用官方 HF。

---

## 环境要点(踩坑前置)

1. **COCO 官方图床 `images.cocodataset.org` 直接 HTTPS 会 TLS 报错**
   - 现象:`curl https://images.cocodataset.org/...` → `SSL: no alternative certificate subject name matches target host name`。
   - 根因:该域名 CNAME 到 S3 virtual-hosted bucket,服务端证书 CN 是 `s3.amazonaws.com`,与带点的 bucket 名不匹配(SNI/证书不匹配,非网络不通)。
   - **修复:改用 S3 path-style URL**(证书匹配 `s3.amazonaws.com`,干净、无需 `-k`):
     ```
     https://s3.amazonaws.com/images.cocodataset.org/val2014/COCO_val2014_<12位id>.jpg
     https://s3.amazonaws.com/images.cocodataset.org/train2014/COCO_train2014_<12位id>.jpg
     ```
   - 备选:`curl -k`(忽略证书)或 `http://`(明文)也能取到,但 path-style 最规范。
   - 实测单图速度 ≈ **17–68 KB/s**,单张 70–380 KB,**单图 <5s**;POPE 仅需 500 张、LLaVA subset 数千张,可接受。

2. `conda run -n dl python - <<EOF` 方式的 heredoc 在本机会吞掉 stdout。**请用绝对路径解释器** `/home/l/miniforge3/envs/dl/bin/python 脚本.py`。

3. 已装:`huggingface_hub 1.23.0`、`pyarrow 25.0.0`、`pandas 3.0.3`、`PIL 12.2.0`。**未装 `datasets`**——读 parquet 直接用 pyarrow 即可,不必装 datasets。

---

## A. 阶段2:视觉指令微调数据(LoRA SFT)

### 推荐源:`liuhaotian/LLaVA-Instruct-150K`(HF dataset,LLaVA 本尊)

- URL:https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K
- **格式实测 = 完全命中硬要求**:JSON 数组,每条
  `{"id", "image", "conversations":[{"from":"human","value":...},{"from":"gpt","value":...}, ...]}`。
  可直接对 `from=="gpt"` 的 turn 算 loss、`from=="human"` 的 turn mask 掉。
- **图像不内嵌**:`image` 字段是裸 COCO id(如 `000000442786.jpg`),对应 **COCO train2014**;需按上文 S3 path-style 另取。已实测这些 id 均能取到真图(见下)。

#### 各文件规模(HF API 实测)与用途

| 文件 | 大小 | 对话结构 | 说明 |
|---|---|---|---|
| `detail_23k.json` | 20.5 MB | **单轮**(23240 条全为 1 问 1 答) | 详细描述,最轻,**format 样本已下载验证** |
| `complex_reasoning_77k.json` | 79.6 MB | 单轮 | 复杂推理 QA |
| `conversation_58k.json` | 126.1 MB | **多轮**(样本实测 6 turns = 3 轮问答) | **真多轮对话**,SFT 多轮首选 |
| `llava_instruct_80k.json` | 130.7 MB | 混合(含多轮) | 150k 的精简版 |
| `llava_instruct_150k.json` | 228.9 MB | 混合(含多轮) | detail+reasoning+conversation 合集,全 COCO train2014 |
| `llava_v1_5_mix665k.json` | **1029.9 MB** | 混合 | v1.5 完整 SFT 集;**>1GB 不 bulk**,且图跨 coco/gqa/ocr_vqa/textvqa/vg 多源,组装麻烦 |

#### 精确下载命令(推荐:多轮用 conversation_58k;仅演示/最快用 detail_23k)

```bash
BASE=https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main
# 最轻(20.5MB,实测 141s @145KB/s 全量下完):
curl -L -o data/sft/detail_23k.json          "$BASE/detail_23k.json"
# 真多轮(126MB,约 14 分钟):
curl -L -o data/sft/conversation_58k.json     "$BASE/conversation_58k.json"
# 若要标准全量(228MB,约 26 分钟):
curl -L -o data/sft/llava_instruct_150k.json  "$BASE/llava_instruct_150k.json"
```

演示只需**几千~几万条 subset**:下 `conversation_58k.json` 后 `json.load` 取前 N 条即可,不必全量。

#### 配图下载(subset 图,按需逐张取,别整包拉 train2014 13GB)

```bash
# image 字段 "000000442786.jpg" -> train2014
id=000000442786
curl -L -o data/sft/images/COCO_train2014_${id}.jpg \
  "https://s3.amazonaws.com/images.cocodataset.org/train2014/COCO_train2014_${id}.jpg"
```

#### 实测格式样本(本机真实打印)

`detail_23k.json`(单轮):
```json
{
  "id": "000000442786",
  "image": "000000442786.jpg",
  "conversations": [
    {"from": "human", "value": "What do you see happening in this image?\n<image>"},
    {"from": "gpt",   "value": "The scene depicts a lively plaza area with several people walking ..."}
  ]
}
```
`conversation_58k.json`(多轮,6 turns):
```
human: What are the colors of the bus in the image?\n<image>
gpt  : The bus in the image is white and red.
human: What feature can be seen on the back of the bus?
gpt  : The back of the bus features an advertisement.
human: Is the bus driving down the street or pulled off to the side?
gpt  : The bus is driving down the street, which is crowded ...
```

- 实测:`detail_23k` 全部 23240 条都有 `image` 字段;`conversations` 是标准 from/value list。
- 图像 id→train2014 已验证可取:`000000442786`→149879B、`000000539056`→133404B、`000000131697`→116230B,均 `code=200`。

#### 注意事项 / 坑

- **`<image>` 占位符位置不固定**:detail 集在 human value **末尾**(`...image?\n<image>`),conversation 集在**首问句里**。LLaVA 官方预处理会把 `<image>` 抽出并统一放到首 turn 开头。接入本项目 `data/dataset.py`(现拼 `<image>\n{question}`)时,**务必先 strip 掉 value 里的 `<image>\n` 再按自己模板重排**,否则会出现双 `<image>` 或位置错乱。
- SFT loss 只在 `from=="gpt"` 的 turn 上算(human turn label=-100),多轮时每个 gpt turn 都要算。
- 图另取有单点风险(S3 图床),见下方降级阶梯的 embedded-image 兜底。

### 降级阶梯(A:试过什么 / 可达性 / 速度 / 取舍)

| 级别 | 候选 | 可达性 / 速度 | 取舍结论 |
|---|---|---|---|
| 0(最标准全量) | `llava_v1_5_mix665k.json` | HF 可达,但 **1030MB >1GB**;图跨 5 个源 | **不用**:违反「不 bulk」,配图组装成本高 |
| 1(标准全量) | `llava_instruct_150k.json` 228MB | 可达,约 26 min;图全 COCO train2014(易取) | 可选,量偏大 |
| 2(**多轮推荐**) | `conversation_58k.json` 126MB | 可达,约 14 min;图 train2014 | **推荐(多轮)**:格式=本尊,图可取 |
| 3(**最轻推荐**) | `detail_23k.json` 20.5MB | **实测 141s @145KB/s 下完**;图 train2014 | **推荐(演示/最快)**:单轮,格式合规 |
| 4(图内嵌兜底) | `HuggingFaceH4/llava-instruct-mix-vsft`(parquet,图嵌 bytes) | HF API 实测:train 分 20 shard,**每片 ≈540MB**(全量 ≈10.6GB) | **仅当 COCO S3 图床不可用时兜底**:1 片 ≈60 min 得约 1.3 万条含图样本;字段是 `messages`(role/content),需转成 `conversations`(from/value) |

> 结论:**COCO S3 图床本机已验证可取**,故优先「JSON(本尊格式)+ S3 逐图」;仅当图床失效才降级到 embedded-image parquet 并做 `messages→conversations` 转换。

---

## B. POPE 幻觉评测

### 推荐源:官方 `RUCAIBox/POPE`(GitHub raw)

- Repo:https://github.com/RUCAIBox/POPE
- 题集 raw URL(三 split,object-existence Yes/No):
  ```
  https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_random.json
  https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_popular.json
  https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_adversarial.json
  ```

#### 下载命令 + 实测

```bash
mkdir -p benchmark/pope
for s in random popular adversarial; do
  curl -L -o benchmark/pope/coco_pope_$s.json \
    "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_$s.json"
done
```
实测:三个文件均 `code=200`,各 ≈370 KB,速度 **100–138 KB/s**,秒下。

#### 格式(JSONL,本机实测)

```json
{"question_id": 1, "image": "COCO_val2014_000000310196.jpg", "text": "Is there a snowboard in the image?", "label": "yes"}
```
- 每 split **3000 条 / 500 张唯一图 / yes:no = 1500:1500**(完美平衡,三 split 一致)。
- 字段:`question_id, image, text, label(yes/no)`。图为 **COCO val2014**。

#### 配图(val2014,按 URL 单张取,已实测)

```bash
# POPE 引用的图,例:
curl -L -o benchmark/pope/images/COCO_val2014_000000310196.jpg \
  "https://s3.amazonaws.com/images.cocodataset.org/val2014/COCO_val2014_000000310196.jpg"
```
实测 POPE 图 `COCO_val2014_000000310196.jpg` → `code=200`、89261B、37 KB/s。500 张唯一图逐张取即可,总量小。

#### 注意事项

- 直接 `https://images.cocodataset.org/...` 会 TLS 报错,**必须走 S3 path-style**(见环境要点1)。
- 评测协议:prompt 用图 + question,模型输出映射到 Yes/No;POPE 报 Accuracy/Precision/Recall/F1,重点看 **F1 与 "Yes" 比例**(幻觉倾向)。三 split 难度递增(random<popular<adversarial)。

### 降级阶梯(B)

| 级别 | 候选 | 可达性 / 速度 | 结论 |
|---|---|---|---|
| 0(官方) | `RUCAIBox/POPE` GitHub raw | 实测 100–138 KB/s,秒下 | **推荐**,题集即用 |
| 1(镜像) | HF 上 `lmms-lab/POPE` 等镜像 | 未测(官方已够快) | 备用,若 GitHub raw 被墙再切 |
| 图 | COCO val2014 via S3 path-style | 实测 37 KB/s、单图 <3s | **推荐**,500 张够用 |

---

## C. Qwen3-0.6B 模型

### 确认:`Qwen/Qwen3-0.6B`(HF)

- URL:https://huggingface.co/Qwen/Qwen3-0.6B
- **权重大小实测**:`model.safetensors` = **1,503,300,328 B ≈ 1.4 GB**(bf16 单文件)。>1GB,本次只探测未拉全量。
- config 实测(`config.json`):`Qwen3ForCausalLM`,`hidden_size=1024`,`num_hidden_layers=28`,`num_attention_heads=16`,`num_key_value_heads=8`(GQA),`vocab_size=151936`,`max_position_embeddings=40960`,`torch_dtype=bfloat16`,`tie_word_embeddings=true`,`transformers>=4.51.0`。

#### 小文件下载(验证用,已实测下完)

```bash
BASE=https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main
for f in config.json tokenizer_config.json generation_config.json tokenizer.json; do
  curl -L -o weights/qwen3-0.6b/$f "$BASE/$f"
done
# 全量权重(后续执行步,约 1.4GB):
# hf download Qwen/Qwen3-0.6B --local-dir weights/qwen3-0.6b
```

### Chat template / thinking mode 兼容性(**已下 tokenizer_config.json 实测**)

- **原生带 chat_template**(len 4168),标准 **ChatML**:`<|im_start|>{role}\n{content}<|im_end|>\n`。
- 特殊 token:`eos = <|im_end|>`(id 151645)、`pad = <|endoftext|>`、无 `bos`。`generation_config` 里 eos 是 `[151645, 151643]`。
- **Thinking mode 确认存在**:chat_template 含 `<think>` 逻辑,支持 `enable_thinking` 开关:
  - `add_generation_prompt=True` 且 `enable_thinking=False` → 模板自动注入空的 `<think>\n\n</think>\n\n`(即关闭推理)。
  - 默认(不传或 True)→ 模型可自由输出 `<think>...</think>` 推理段。

#### 与本项目训练格式的兼容结论(关键)

- **好消息:drop-in 兼容**。本项目 `data/dataset.py` 现已用 ChatML(`<|im_start|>user\n<image>\n{q}<|im_end|>\n<|im_start|>assistant\n` + `{answer}<|im_end|>`),Qwen3 正是这套 ChatML,`eos=<|im_end|>` 也一致,**无需改模板结构**。
- **必做:SFT 时关掉 thinking**。VLM captioning/VQA 不需要推理段,且 LLaVA 数据的 gpt turn 里没有 `<think>`。应在 apply chat template 时传 `enable_thinking=False`(或手动在 assistant 起始拼空 `<think>\n\n</think>\n\n`),否则训练目标与推理格式不一致、且浪费 token。
- **注意 eos**:训练/推理都用 `<|im_end|>`(151645)作为 turn 结束;`generation_config` 把 `<|endoftext|>`(151643)也列为 eos,自定义生成循环时两者都要当停止符。
- 现仓库用的是 `Qwen2.5-0.5B`(`weights/weights/Qwen--Qwen2.5-0.5B`),同为 ChatML + `<|im_end|>`,换到 Qwen3-0.6B 只需增加 thinking 关闭处理。

### 降级阶梯(C)

| 级别 | 候选 | 状态 | 结论 |
|---|---|---|---|
| 0 | `Qwen/Qwen3-0.6B` | config/tokenizer 实测已下,权重 1.4GB 已探测 | **推荐**,ChatML 原生兼容 |
| 1(镜像) | `hf-mirror.com/Qwen/Qwen3-0.6B` 或 ModelScope `Qwen/Qwen3-0.6B` | hf-mirror 实测与直连同速(~160KB/s) | 若直连拉权重慢可切镜像,格式完全一致 |
| 2(已有) | `Qwen/Qwen2.5-0.5B`(仓库现用) | 本地已有 | 兜底,同 ChatML,无 thinking 负担 |

---

## 三样最终推荐(一句话)

- **A 视觉指令微调**:`liuhaotian/LLaVA-Instruct-150K` — 多轮用 `conversation_58k.json`(126MB,~14min),演示最快用 `detail_23k.json`(实测 141s 下完);格式即 `conversations` from/value 本尊,图走 COCO train2014 S3 path-style 逐张取(已验证)。
- **B POPE**:官方 `RUCAIBox/POPE` GitHub raw 三 split(各 3000 题/500 图,秒下),图走 COCO val2014 S3 path-style(已验证)。
- **C 模型**:`Qwen/Qwen3-0.6B`(权重 1.4GB),原生 ChatML 与本项目训练格式 drop-in 兼容,**SFT 需 `enable_thinking=False` 关掉 thinking**。

**卡点**:唯一需注意的是 COCO 官方图床直连 TLS 证书不匹配,**必须改用 `https://s3.amazonaws.com/images.cocodataset.org/...` path-style**(已给修复)。除此之外全部实测可下,无阻塞;整体吞吐 ~145–175 KB/s,大文件(mix665k 1GB、权重 1.4GB)较慢,按需分片、勿整包拉。
