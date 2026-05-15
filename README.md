# AlignSQL

> 基于 Qwen3-8B 的 NL2SQL 全流程微调，从 SFT 到 DPO 的完整对齐实践。

AlignSQL 以 NL2SQL 为切入点，完整跑通大语言模型微调的 **SFT + DPO 全链路**，覆盖数据处理、LoRA 微调、偏好对自动构建、执行准确率评估等环节。

## 整体流程

```
Spider 数据 → 数据预处理 → SFT 训练 → 候选 SQL 生成
                                  ↓              ↓
                             LLaMA-Factory  SQLite 执行验证
                                  ↓              ↓
                              SFT 模型        偏好对构建
                                  ↓              ↓
                               └────→ DPO 训练 ←─┘
```

## 快速开始

### 1. 数据准备

```bash
# 下载 Spider 数据集（HF 镜像）
uv run python scripts/download_spider.py
```

数据处理脚本参考 `scripts/prepare_sft.py`，将 Spider 原始 JSON 转为 LLaMA-Factory 兼容的 Alpaca 格式。

### 2. SFT 训练

```bash
llamafactory-cli train config/sft.yaml
```

核心配置：Qwen3-8B + LoRA (rank=32) + 梯度累积 ×8，单卡 RTX 4090 上约 3-4 小时。

### 3. DPO 偏好对齐

先执行反馈构建偏好对：

```bash
python scripts/generate_candidates.py
python scripts/build_preferences.py
```

再启动 DPO 训练：

```bash
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
├── config/                  # 训练配置文件
│   ├── sft.yaml             # SFT 训练配置（LoRA、学习率、epoch）
│   └── dpo.yaml             # DPO 训练配置（β、学习率）
│
├── scripts/                 # 数据处理与评估脚本
│   ├── prepare_sft.py       # Spider → Alpaca 格式转换
│   ├── generate_candidates.py  # SFT 模型生成候选 SQL
│   ├── build_preferences.py    # 执行反馈构建 DPO 偏好对
│   └── evaluate.py          # 执行准确率评估
│
├── models/                  # 训练输出目录（gitignore）
│   ├── sft/                 # SFT LoRA adapter 权重
│   └── dpo/                 # DPO LoRA adapter 权重
│
├── experiments/             # 实验日志与评估结果
│   └── logs/
│
├── docs/                    # 详细方案文档
│   └── project_report.md    # 完整技术方案文档
│
└── README.md                # 项目入口（本文）
```

## 详细方案

项目技术方案文档见 [docs/project_report.md](docs/project_report.md)，包含：

- 数据准备与 Schema 序列化设计
- SFT 与 DPO 训练配置详解
- 偏好对自动构建逻辑
- 评估方法与预期结果
- 实验追踪配置（wandb）

## License

MIT
