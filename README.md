# AlignSQL

> 基于 Qwen3-8B 的 NL2SQL 全流程微调，从 SFT 到 DPO 的完整对齐实践。

AlignSQL 以 NL2SQL 为切入点，完整跑通大语言模型微调的 **SFT + DPO 全链路**，覆盖数据处理、LoRA 微调、偏好对自动构建、执行准确率评估等环节。

## Spider 数据集难度分布

Spider 数据集按 SQL 复杂度分为 4 个难度级别：

| 难度 | 占比 | SQL 特征 | 示例 |
|------|:----:|----------|------|
| Easy | 31.7% | 简单 SELECT + WHERE | `SELECT name FROM users WHERE age > 18` |
| Medium | 53.7% | 聚合/GROUP BY/ORDER BY | `SELECT count(*) FROM head WHERE age > 56` |
| Hard | 7.0% | 多表 JOIN / 子查询 | `SELECT * FROM A WHERE id NOT IN (SELECT id FROM B)` |
| Extra | 7.5% | UNION / INTERSECT | `SELECT name FROM A INTERSECT SELECT name FROM B` |

使用 `scripts/analyze_difficulty.py` 分析数据集难度分布：
```bash
python scripts/analyze_difficulty.py -i dataset/train-00000-of-00001.parquet
```

## 整体流程

> 待完善

## 快速开始

### Phase 1: Zero-shot Baseline

```bash
# 测试 Qwen3-8B 基座模型的零样本能力
python scripts/evaluate.py \
    --model_path /path/to/Qwen3-8B \
    --spider_dir /path/to/spider \
    --stage zeroshot
```

### Phase 2: SFT 训练

```bash
# 数据预处理：Spider → Alpaca 格式
python scripts/prepare_sft.py

# 启动 SFT 训练
llamafactory-cli train config/sft.yaml
```

核心配置：Qwen3-8B + LoRA (rank=32) + 梯度累积 ×8，单卡 RTX 4090 上约 3-4 小时。

### Phase 3: DPO 偏好对齐

```bash
# SFT 模型生成候选 SQL
python scripts/generate_candidates.py

# 执行反馈构建偏好对
python scripts/build_preferences.py

# 启动 DPO 训练
llamafactory-cli train config/dpo.yaml
```

## 实验结果

| 阶段 | Spider (EX) | 说明 |
|------|:-----------:|------|
| Zero-shot (baseline) | ~55% | Qwen3-8B 直接 prompt |
| SFT 后 | ~78% | 标准 LoRA 微调 |
| DPO 后 | ~82% | 执行反馈偏好对齐 |

DPO 的主要增益集中在 Hard / Extra Hard 级别的复杂 SQL 上。

## 技术栈

- **基座模型**：Qwen3-8B
- **微调框架**：LLaMA-Factory (LoRA)
- **数据集**：Spider
- **实验追踪**：Weights & Biases
- **硬件**：RTX 4090 (24GB) × 1

## 项目结构

```
AlignSQL/
├── config/                     # 配置文件
│   ├── sft.yaml               # SFT 训练配置
│   └── dpo.yaml               # DPO 训练配置
├── data/                      # 数据目录
│   ├── spider/                # Spider 原始数据
│   │   ├── train.json
│   │   ├── dev.json
│   │   ├── tables.json
│   │   └── database/         # SQLite 数据库文件
│   ├── sft/                   # SFT 训练数据（LLaMA-Factory 格式）
│   └── dpo/                   # DPO 偏好数据
├── scripts/                   # 脚本
│   ├── prepare_sft.py        # Spider → Alpaca 格式
│   ├── analyze_difficulty.py # 数据集难度分析
│   ├── generate_candidates.py # SFT 模型生成候选 SQL
│   ├── build_preferences.py  # 执行反馈构建偏好对
│   ├── evaluate_vllm.py      # 统一评测脚本（--stage 区分）
│   └── run_zeroshot.sh       # 批量运行脚本
├── models/                    # 模型权重输出
│   ├── sft/                  # SFT adapter 权重
│   └── dpo/                  # DPO adapter 权重
├── experiments/               # 实验结果
│   ├── zeroshot/             # Zero-shot 结果
│   │   └── results.json
│   ├── sft/                  # SFT 结果
│   │   └── results.json
│   └── dpo/                  # DPO 结果
│       └── results.json
└── README.md
```

## 详细方案

项目技术方案文档见 [docs/project_report.md](docs/project_report.md)，包含：

- 数据准备与 Schema 序列化设计
- SFT 与 DPO 训练配置详解
- 偏好对自动构建逻辑
- 评估方法与预期结果
- 实验追踪配置（wandb）

## License

[MIT](LICENSE)

## 致谢

- [Qwen3](https://github.com/QwenLM/Qwen3) — 基座模型
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — 微调框架
- [Spider](https://yale-lily.github.io/spider) — 数据集
- [Weights & Biases](https://wandb.ai) — 实验追踪
- [DB-GPT-Hub](https://github.com/eosphoros-ai/DB-GPT-Hub) — 方案参考