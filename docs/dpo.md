# AlignSQL DPO 偏好对齐

> 基于 SFT 模型采样 + 局部错误分类构造 preference pairs，对 Qwen3-8B 进行 DPO 训练。

## 概述

DPO 训练的目标是让模型学会**偏好正确的 SQL 生成行为**。核心思路：

1. 用 SFT 模型对每条训练问题采样多个候选 SQL
2. 通过执行对比，识别哪些候选是错误的（rejected）
3. 以 gold SQL 为主要 chosen，并补充通过质量门控的执行正确候选
4. 按错误类型平衡数据，训练模型学会避开常见错误模式

```
SFT 模型 (Qwen3-8B)
    ↓ 采样候选 (T=1.2, SC-N)
    ↓ 执行对比 + 错误分类
    ↓ 构造 preference pairs
DPO 训练 (LLaMA-Factory)
    ↓
DPO 模型 → EX 74.0% (Greedy) / 75.0% (SC-12)
```

## 设计思路: 从 SFT 的不足出发

### SFT 的局限性

SFT 模型在 Spider dev 上达到 71.8% EX，但错误分析发现：

| 错误类型 | 占比 | 典型表现 |
|---------|:----:|----------|
| TABLE_SELECTION | ~30% | 选错表/多表或少表 |
| SYNTAX_ERROR | ~15% | SQL 语法错误 |
| GROUP_BY | ~15% | 缺少或多余的 GROUP BY |
| WHERE | ~15% | 条件写错 |
| DISTINCT / AGGREGATION | ~15% | DISTINCT 遗漏或聚合函数用错 |
| SET_OP / SUBQUERY | ~5% | UNION/子查询结构错误 |

这说明 SFT 模型虽然整体能力不错，但在**局部细节上仍然不稳定**——同样的 query，采样多次可能得到不同质量的 SQL。

### DPO 数据构造策略

基于上述分析，我们设计了 **local hard-negative 策略**：

1. **chosen 以 gold SQL 为主，门控 sampled-correct 为辅**：每个问题优先构造 `gold > rejected` 的 pair；如果模型采样出的 SQL 执行结果正确、非空、与 gold 在结构和 token 上足够接近，也可作为 `sampled_correct > rejected` 的补充 pair。未经门控的 sampled-correct 风险较高，可能只是当前数据库状态下偶然正确。

2. **rejected 按错误类型分级**：对每个错误候选，通过局部特征对比自动分类（`classify_local_error`），优先选择与 gold SQL 结构相近但执行结果不同的 hard negatives。

3. **错误类型平衡**：按预设比例（`DEFAULT_ERROR_MIX`）从各错误类型中采样，确保训练数据覆盖均匀，避免模型只学会避免某一种错误。

## 数据准备流程

```bash
# 方式一：从 SFT 模型采样候选
python scripts/prepare_dpo.py \
    --model_path /root/autodl-tmp/models/sft/merged \
    --spider_dir dataset/spider \
    --output data_processed/dpo_pairs.json

# 方式二：复用 SC 候选（跳过模型加载）
python scripts/prepare_dpo.py \
    --candidates_file outputs/spider/sft/sc_n8/candidates.json \
    --spider_dir dataset/spider \
    --output data_processed/dpo_pairs.json

# 方式三：含更多变异负样本 + 多温度采样
python scripts/prepare_dpo.py \
    --model_path /root/autodl-tmp/models/sft/merged \
    --n_candidates 8 \
    --temperature 1.2 \
    --extra_temperatures 0.8,1.6 \
    --mutation_negatives 4 \
    --max_pairs_per_question 4
```

### 流程详解

```
加载训练数据 (train_spider.json ± train_others.json)
    ↓
过滤无效问题（缺 schema / 缺 DB / gold SQL 执行失败）
    ↓
对每条问题采样 N 个候选 SQL
    ↓
在 SQLite 上执行每个候选，与 gold 结果对比
    ↓
分类错误类型，过滤低质量文本
    ↓
构造 preference pairs
    ↓
按错误类型平衡（可选）
    ↓
保存为 LLaMA-Factory DPO 格式
```

### 候选采样参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_candidates` | 8 | 每轮候选数 |
| `temperature` | 1.2 | 采样温度（高 = 更多样） |
| `extra_temperatures` | — | 附加温度，如 `0.8,1.6` |
| `candidate_rounds` | 1 | 每温度的采样轮次 |
| `max_pairs_per_question` | 2 | 最多保留的 pairs 数 |

### 质量过滤

采样得到的候选 SQL 在执行前经过 `sql_text_quality` 过滤：

- **非 SQL 开头**：不以 SELECT/WITH 等关键词开头
- **噪声标记**：含 `###REQUEST`、`` ``` `` 等注入痕迹
- **非 ASCII 污染**：含过多非英文字符（多语种乱码）
- **控制字符**：含非法控制字符

第一轮数据中曾发现 ~2.7% 的候选包含多语种 Web 语料污染，这些在后续版本中已被过滤。

### 局部错误分类

对每个执行结果错误的候选，通过特征对比自动分类：

| 错误类型 | 判断依据 |
|---------|----------|
| TABLE_SELECTION | 使用的表集合与 gold 不同 |
| GROUP_BY | GROUP BY 子句缺失或不同 |
| WHERE | WHERE 条件不同 |
| DISTINCT | DISTINCT 有无不一致 |
| AGGREGATION | 聚合函数不同 |
| SET_OP | UNION/INTERSECT/EXCEPT 不同 |
| SUBQUERY | 子查询数量差异 >1 |
| ORDER_BY_LIMIT | ORDER BY / LIMIT 差异 |
| SELECT_COLUMNS | SELECT 列数不同 |
| VALUE_OR_COLUMN | 结构相同但值不同 |
| SYNTAX_ERROR | SQL 执行语法错误 |
| SCHEMA_ERROR | 表或列不存在 |
| EXECUTION_ERROR | 其他执行时错误 |

优先级排序：`TABLE_SELECTION > GROUP_BY > WHERE > DISTINCT > AGGREGATION > SET_OP > SUBQUERY > ORDER_BY_LIMIT > SYNTAX_ERROR > SCHEMA_ERROR > SELECT_COLUMNS > ...`

同类错误中，优先选择与 gold SQL 相似度最高（token similarity）的候选作为 hard negative。

### 变异负样本

除了模型采样的错误候选项，还支持通过 `--mutation_negatives` 从 gold SQL 生成有监督的变异负样本：

```python
# 规则示例
ORDER BY → 删除
LIMIT n → 删除
DISTINCT → 删除
ASC → DESC
COUNT → SUM
MAX → MIN
>= → <
```

这些变异在执行验证后加入 rejected 池，确保那些模型不太容易犯的「低级错误」也有覆盖。

## 数据格式

输出为 LLaMA-Factory DPO 格式（messages-based）：

```json
{
  "messages": [
    {"role": "system", "content": "I want you to act as a SQL terminal..."},
    {"role": "user", "content": "Convert the following question to a SQL query...\n\n###Input:\n{Schema}\n\nQuestion: {question}\n\n###Response:"}
  ],
  "chosen": "SELECT T1.flno FROM Flight AS T1 JOIN Aircraft AS T2 ON T1.aid = T2.aid WHERE T2.name = 'Airbus A340-300'",
  "chosen_source": "gold",
  "rejected": "SELECT flno FROM Flight WHERE aid = (SELECT aid FROM Aircraft WHERE name = 'Airbus A340-300')",
  "db_id": "flight_1",
  "question": "What are the flight numbers for the aircraft Airbus A340-300?",
  "gold_sql": "SELECT T1.flno FROM Flight AS T1 JOIN Aircraft AS T2 ON T1.aid = T2.aid WHERE T2.name = 'Airbus A340-300'",
  "difficulty": "easy",
  "error_type": "TABLE_SELECTION",
  "error_details": ["tables gold=['aircraft', 'flight'], rejected=['aircraft', 'flight']"],
  "candidate_exec_success": true,
  "similarity": 0.75,
  "negative_source": "sampled"
}
```

### 字段说明

| 字段 | 说明 |
|------|------|
| `messages` | SFT prompt 格式，与训练时的输入一致 |
| `chosen` | 偏好的 SQL（通常为 gold SQL） |
| `chosen_source` | chosen 来源：`gold` / `sampled_correct` |
| `rejected` | 非偏好的 SQL（错误候选） |
| `error_type` | 局部错误分类 |
| `error_details` | 错误分析详情 |
| `similarity` | rejected 与 gold SQL 的 token 级相似度 |
| `negative_source` | rejected 来源：`sampled` / `mutation` |

## 训练

### LLaMA-Factory 配置

```yaml
### model
model_name_or_path: /path/to/sft/merged
trust_remote_code: true

### method
stage: dpo
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0
lora_target: all

### dataset
dataset: spider_dpo
dataset_dir: data
template: qwen3_nothink
cutoff_len: 2048
preprocessing_num_workers: 16

### dpo
pref_beta: 0.1
pref_loss: sigmoid

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5e-6
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
flash_attn: auto
optim: adamw_torch

### output
output_dir: models/dpo/qwen3-8b-spider
logging_steps: 5
save_steps: 100

### wandb
report_to: wandb
```

### 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `pref_beta` | 0.1 | DPO β 参数，控制 KL 惩罚强度 |
| `pref_loss` | sigmoid | 标准 DPO loss |
| `learning_rate` | 5e-6 | 比 SFT 低（DPO 更敏感） |
| `num_train_epochs` | 3 | 训练轮次 |

## 实验结果

### Spider Dev（全量 1034 样本）

| 阶段 | Execution | Exact Match |
|------|:---------:|:-----------:|
| Zero-shot (baseline) | 43.9% | 35.7% |
| SFT (Greedy) | 71.8% | 67.0% |
| **DPO (Greedy)** | **74.0%** | **68.1%** |
| **DPO + SC-5** | **74.6%** | **68.2%** |
| **DPO + SC-8** | **74.5%** | **68.0%** |
| **DPO + SC-12** | **75.0%** | **68.9%** |

### 按难度分级（DPO Greedy）

| 难度 | 样本数 | Execution | Exact Match |
|------|:------:|:---------:|:-----------:|
| easy | 248 | 88.7% | 83.9% |
| medium | 446 | 76.0% | 73.8% |
| hard | 174 | 67.2% | 56.3% |
| extra | 166 | 53.6% | 41.6% |
| **all** | **1034** | **74.0%** | **68.1%** |

### DPO vs SFT 提升

| 难度 | SFT EX | DPO EX | 提升 |
|------|:------:|:------:|:----:|
| easy | 89.1% | 88.7% | -0.4% |
| medium | 74.4% | 76.0% | +1.6% |
| hard | 65.5% | 67.2% | +1.7% |
| extra | 48.2% | 53.6% | +5.4% |
| **all** | **71.8%** | **74.0%** | **+2.2%** |

DPO 提升主要集中在 medium 以上难度，其中 extra 提升最显著（+5.4%），说明偏好对齐在复杂 SQL 生成上的收益更大。

### DPO vs SFT error-type breakdown

| 错误类型 | SFT | DPO | 变化 |
|---------|:---:|:---:|:----:|
| TABLE_SELECTION | 30.9% | 26.7% | **-4.2%** |
| SYNTAX_ERROR | 12.3% | 10.8% | -1.5% |
| GROUP_BY | 8.3% | 7.5% | -0.8% |
| WHERE | 9.4% | 7.8% | -1.6% |
| AGGREGATION | 4.9% | 5.2% | +0.3% |

TABLE_SELECTION 改善最明显，说明 DPO 有效减少了模型选错表的倾向。

## 经验教训

### 数据质量优先

首轮 DPO 训练后 Spider EX 仅 71.95%，几乎没有提升。排查发现：

- **未经门控的 sampled-correct chosen 风险较高**：执行正确但非 gold 的 SQL 可能只是当前数据库状态下偶然正确。如果不做相似度、结构和结果质量门控，模型会学习到不稳定偏好。
- **103 对 chosen/rejected 完全一致**：去重逻辑不足，模型从中学不到任何偏好信号。

**教训**：DPO 数据构造后必须验证 chosen 来源分布、chosen/rejected divergence、SQL 文本噪声，以及 sampled-correct chosen 是否通过质量门控。`chosen != gold_sql` 并非一定不可用，但必须谨慎过滤。

### 训练指标不反映数据质量

首轮训练中 `reward_accuracy` 等指标表现正常，并未提示数据有问题。训练指标只反映「偏好是否被优化」，不反映「偏好是否正确」。数据质量必须直接 inspect。

### Prompt 一致性

DPO 训练和评测使用不同 prompt 格式会导致分数虚低。本项目中训练和评测统一使用 SFT prompt 格式，避免不一致问题。

## 进阶用法

### 多温度采样

```bash
python scripts/prepare_dpo.py \
    --model_path /root/autodl-tmp/models/sft/merged \
    --n_candidates 5 \
    --temperature 1.2 \
    --extra_temperatures 0.8,1.6 \
    --candidate_rounds 2
```

多温度 + 多轮次可以大幅增加候选多样性，但同时也增加计算成本。建议先从 `--candidate_rounds 1` 开始。

### 自定义错误类型混合

```bash
python scripts/prepare_dpo.py \
    --model_path /path/to/model \
    --error_mix "TABLE_SELECTION:0.3,SYNTAX_ERROR:0.2,GROUP_BY:0.2,WHERE:0.15,AGGREGATION:0.15" \
    --target_pairs 6000
```

通过 `--error_mix` 可以自定义各错误类型的目标比例，`--target_pairs` 控制最终输出的 pair 数量。

## 参考

- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)
- [LLaMA-Factory DPO 文档](https://github.com/hiyouga/LLaMA-Factory)
- [SFT 训练流程](sft.md)
