# AlignSQL SFT 训练流程及评测方案

> 基于 Spider 数据集的 NL2SQL 微调完整流程

## 概述

AlignSQL 使用 LLaMA-Factory 对 Qwen3-8B 进行 SFT 训练，实现 Text-to-SQL 任务。

```
Qwen3-8B (基座)
    ↓ SFT
AlignSQL (微调模型) → EX 准确率 72.24%
```

## 数据集

### Spider 数据集

| 文件 | 数量 | 用途 |
|------|------|------|
| `train_spider.json` | 7,000 | 训练集 |
| `train_others.json` | 1,659 | 训练补充 |
| `dev.json` | 1,034 | 本地验证 |
| `test.json` | 2,147 | 最终测试 |
| `tables.json` | 166 个 DB | Schema 定义 |

**数据来源**: [Spider](https://yale-lily.github.io/spider) (CC BY-SA 4.0)

### 数据集难度分布

| 难度 | 占比 | SQL 特征 |
|------|:----:|----------|
| Easy | 31.7% | 简单 SELECT + WHERE |
| Medium | 53.7% | 聚合/GROUP BY/ORDER BY |
| Hard | 7.0% | 多表 JOIN / 子查询 |
| Extra | 7.5% | UNION / INTERSECT |

## 流程详解

### Step 1: 数据预处理

```bash
python scripts/prepare_sft.py
```

**输入**:
- `dataset/train_spider.json` - 原始问题 SQL 对
- `dataset/tables.json` - 数据库 Schema

**输出**:
- `data/sft_data.json` - LLaMA-Factory 格式数据

### 数据格式

预处理后的数据格式为 Alpaca 格式：

```json
{
  "db_id": "department_management",
  "difficulty": "medium",
  "system": "I want you to act as a SQL terminal in front of an example database,...",
  "instruction": "Convert the following question to a SQL query based on the database schema.",
  "input": "###Input:\n{schema}\n\nQuestion: {question}\n\n###Response:",
  "output": "SELECT count(*) FROM head WHERE age > 56"
}
```

### Schema 构建

Schema 从 `tables.json` 转换为自然语言描述：

```
department_management contains tables such as department, head, management.
Table department has columns such as Department_ID, Name, Creation, Ranking.
department.Department_ID is the primary key.
The head_ID of management is the foreign key of head_ID of head.
```

### Step 2: 配置训练参数

编辑 `config/sft.yaml`:

```yaml
### model
model_name_or_path: /path/to/Qwen3-8B
trust_remote_code: true

### method
stage: sft
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0
lora_target: all
enable_thinking: false

### dataset
dataset: spider_sft
dataset_dir: data
template: qwen3_nothink
cutoff_len: 1024
max_samples: 100000
val_size: 0.1
preprocessing_num_workers: 16

### output
output_dir: models/sft/qwen3-8b-spider
logging_steps: 5
save_steps: 100

### train
num_train_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1e-4
lr_scheduler_type: cosine
warmup_steps: 0
max_grad_norm: 1.0
bf16: true
flash_attn: auto
optim: adamw_torch

### wandb
report_to: wandb
```

配置 `config/dataset_info.json`:

```json
{
  "spider_sft": {
    "file_name": "data/sft_data.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system"
    }
  }
}
```

### Step 3: 启动训练

```bash
llamafactory-cli train config/sft.yaml
```

**硬件要求**: RTX 4090 (24GB) × 1
**训练时间**: 约 2 小时

## 评测结果

### Qwen3-8B SFT（Spider dev 全量 1034 样本）

| 指标 | 准确率 | 说明 |
|------|--------|------|
| **Execution Accuracy** | **72.24%** | SQL 执行结果一致 |
| **Exact Match** | **67.41%** | SQL 结构完全匹配 |

### 按难度分级

| 难度 | 样本数 | Execution | Exact Match |
|------|--------|-----------|-------------|
| easy | 248 | 89.11% | 83.06% |
| medium | 446 | 74.44% | 74.22% |
| hard | 174 | 65.52% | 52.87% |
| extra | 166 | 48.19% | 40.96% |
| **all** | **1034** | **72.24%** | **67.41%** |

### exec vs exact 的区别

| 指标 | 说明 |
|------|------|
| **exec** | 把预测 SQL 和标准 SQL 都在真实数据库上执行，结果一致就对 |
| **exact** | 把 SQL 解析成结构，逐 clause 比对（select/where/group by 等），结构完全一致才算对 |

- `exec` 通常 >= `exact`（差值说明语义对但写法不完全一致）
- 两者差约 **5%**，说明模型生成的 SQL 语义正确性较高，与标准答案结构差异较小

### SFT vs Zero-shot 对比

| 难度 | Zero-shot EX | SFT EX | 提升 |
|------|-------------|--------|------|
| easy | 72.18% | 89.11% | +16.93% |
| medium | 45.96% | 74.44% | +28.48% |
| hard | 25.86% | 65.52% | +39.66% |
| extra | 15.06% | 48.19% | +33.13% |
| **all** | **43.91%** | **72.24%** | **+28.33%** |

SFT 微调在所有难度级别上都带来了显著提升，尤其是 hard 和 extra 级别的增幅最大。

## 训练参数说明

### Batch Size 与梯度累积

由于 GPU 显存限制，无法一次性处理过大的 batch，因此使用梯度累积技术：

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `batch_size` | GPU 每步处理的样本数 | 4 |
| `gradient_accumulation_steps` | 累积多少步才更新一次模型 | 4 |
| **有效 batch** | 实际等效的 batch size = batch × accum | **16** |

```
Step 1: 计算梯度 → 保存（不更新）
Step 2: 计算梯度 → 累加
Step 3: 计算梯度 → 累加
Step 4: 计算梯度 → 累加
        ↓ 累积完成，更新模型！
```

| 参数 | 值 | 说明 |
|------|---|------|
| `lora_rank` | 8 | LoRA 秩，越大容量越大 |
| `lora_alpha` | 64 | LoRA 缩放因子，通常 2× rank |
| `lora_target` | all | 作用于所有层 |
| `val_size` | 0.1 | 10% 数据作为验证集 |
| `warmup_ratio` | 0.1 | 预热比例 |
| `cutoff_len` | 2048 | 最大序列长度 |

## 训练输出

训练完成后，模型权重保存在:

```
models/sft/qwen3-8b-spider/
├── adapter_config.json
├── adapter_model.safetensors
└── tokenizer_config.json
```

## 下一步

- [DPO 偏好对齐](./dpo.md) - 预期进一步提升到 ~82%
- [评测文档](./evaluation.md) - 完整评测方法

## 参考

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Spider Dataset](https://yale-lily.github.io/spider)
- [DB-GPT-Hub](https://github.com/eosphoros-ai/DB-GPT-Hub)