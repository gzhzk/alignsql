# AlignSQL：Qwen3-8B 的 NL2SQL 全流程微调

> 从 SFT 到 DPO，完整跑通 NL2SQL 的模型对齐实践。
>
> 基座模型：Qwen3-8B | 数据集：Spider | 框架：LLaMA-Factory | 硬件：RTX 4090 (24GB)

## 整体流程

> 待完善

## 快速开始

### Phase 1: Zero-shot Baseline

```bash
# 测试 Qwen3-8B 基座模型的零样本能力
python scripts/evaluate_vllm.py \
    --model_path /path/to/Qwen3-8B \
    --spider_dir dataset \
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
| Zero-shot (baseline) | ~42% | Qwen3-8B 直接 prompt |
| SFT 后 | ~78% | 标准 LoRA 微调 |
| DPO 后 | ~82% | 执行反馈偏好对齐 |

DPO 的主要增益集中在 Hard / Extra Hard 级别的复杂 SQL 上。

## 技术栈

- **基座模型**：Qwen3-8B
- **微调框架**：LLaMA-Factory (LoRA)
- **推理加速**：vLLM
- **数据集**：Spider
- **实验追踪**：Weights & Biases
- **硬件**：RTX 4090 (24GB) × 1

## Spider 数据集难度分布

Spider 数据集按 SQL 复杂度分为 4 个难度级别：

| 难度 | 占比 | SQL 特征 |
|------|:----:|----------|
| Easy | 31.7% | 简单 SELECT + WHERE |
| Medium | 53.7% | 聚合/GROUP BY/ORDER BY |
| Hard | 7.0% | 多表 JOIN / 子查询 |
| Extra | 7.5% | UNION / INTERSECT |

```bash
# 分析数据集难度分布
python scripts/analyze_difficulty.py -i dataset/train-00000-of-00001.parquet
```

## 数据集

使用 [Spider](https://yale-lily.github.io/spider) 数据集（CC BY-SA 4.0）

```bash
# 下载完整数据集（含数据库）
# 官方链接: https://drive.google.com/uc?id=1TqleXec_OykOYFREKKtschzY29dUcVAQ
# 解压后放到 dataset/ 目录
```

> **注意**：`dataset/database/` 和 `dataset/test_database/` 不包含在 git 仓库中，需要单独下载。

## 项目结构

```
AlignSQL/
├── config/                     # 配置文件
│   ├── sft.yaml               # SFT 训练配置
│   └── dpo.yaml               # DPO 训练配置
├── dataset/                   # 数据目录
│   ├── train-*.parquet        # 训练数据
│   ├── validation-*.parquet   # 验证数据
│   ├── train_spider.json      # 训练集
│   ├── dev.json               # 开发集
│   ├── tables.json            # Schema 定义
│   ├── database/              # SQLite 数据库（需单独下载）
│   └── test_database/         # 测试数据库（需单独下载）
├── scripts/                   # 脚本
│   ├── prepare_sft.py        # 数据预处理
│   ├── analyze_difficulty.py # 难度分析
│   ├── evaluate_vllm.py     # 评测脚本
│   └── ...
├── models/                    # 模型权重输出
├── experiments/               # 实验结果
└── README.md
```

## 详细方案

项目技术方案文档见 [docs/project_report.md](docs/project_report.md)，包含：

- 数据准备与 Schema 序列化设计
- SFT 与 DPO 训练配置详解
- 偏好对自动构建逻辑
- 评估方法与预期结果

## License

[MIT](LICENSE)

## 致谢

- [Qwen3](https://github.com/QwenLM/Qwen3) — 基座模型
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — 微调框架
- [vLLM](https://github.com/vllm-project/vllm) — 推理加速
- [Spider](https://yale-lily.github.io/spider) — 数据集（CC BY-SA 4.0）
- [Weights & Biases](https://wandb.ai) — 实验追踪
- [DB-GPT-Hub](https://github.com/eosphoros-ai/DB-GPT-Hub) — 方案参考