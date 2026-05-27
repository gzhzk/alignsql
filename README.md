# AlignSQL：NL2SQL 全流程 Post-training Pipeline

> 从 SFT 到 DPO 再到 RL，完整跑通 NL2SQL 的模型对齐实践。
>
> 基座模型：Qwen3-8B | 框架：LLaMA-Factory | 硬件：RTX 4090 (24GB)

---

## 实验结果

| 阶段 | Spider Dev (EX) | Exact Match | 状态 |
|------|:---------------:|:-----------:|:----:|
| Zero-shot (baseline) | 43.91% | 35.69% | ✅ |
| SFT (LoRA rank=8) | 72.24% | 67.41% | ✅ |

### 按难度分级（SFT）

| 难度 | Zero-shot | SFT | 提升 |
|------|-----------|-----|:----:|
| easy | 72.18% | 89.11% | +16.93% |
| medium | 45.96% | 74.44% | +28.48% |
| hard | 25.86% | 65.52% | +39.66% |
| extra | 15.06% | 48.19% | +33.13% |

## 快速开始

```bash
# Zero-shot 评测
bash scripts/run_zeroshot.sh

# SFT 训练与评测
python scripts/prepare_sft.py
llamafactory-cli train configs/spider/sft.yaml
bash scripts/run_sft.sh

# Self-Consistency 推理 (N=5)
bash scripts/run_sc.sh 5
```

## 项目结构

```
├── alignsql/                      # Python 包 (pip install -e .)
│   ├── __init__.py
│   ├── data/                      # 数据加载/处理
│   │   ├── preprocessing.py       # 难度分类、Prompt 构建
│   │   ├── schema.py              # Schema 序列化
│   │   └── spider.py              # Spider 数据加载器
│   ├── models/                    # 模型训练/推理
│   │   └── inference.py           # Self-Consistency 推理策略
│   ├── analysis/                  # 错误分析、消融对比
│   ├── eval/                      # 评估指标
│   │   └── metrics.py
│   └── utils/                     # 工具函数
│       ├── db.py                  # SQLite 执行工具
│       └── io.py                  # JSON/JSONL 读写
├── vendor/                        # 第三方代码 (Spider 官方评测)
│   ├── evaluation.py
│   └── process_sql.py
├── configs/                       # LLaMA-Factory 训练配置
│   ├── dataset_info.json
│   └── spider/
│       ├── sft.yaml
│       └── merge_sft.yaml
├── scripts/                       # 可执行入口
│   ├── evaluate_vllm.py           # 推理评测 (支持 SC)
│   ├── prepare_sft.py
│   ├── analyze_difficulty.py
│   ├── run_sft.sh
│   ├── run_zeroshot.sh
│   └── run_sc.sh
├── tests/
├── outputs/
├── assets/
│   ├── sft-train-loss.png
│   ├── sft-eval-loss.png
│   └── sft-learning-rate.png
├── docs/
└── setup.py
```

## 详细方案

- [重构计划](docs/PLANNING.md)
- [SFT 训练流程](docs/sft.md)
- [项目报告](docs/project-report.md)

## 技术栈

| 组件 | 选型 |
|------|------|
| 基座模型 | Qwen3-8B |
| 微调框架 | LLaMA-Factory (LoRA) |
| 推理加速 | vLLM |
| 实验追踪 | Weights & Biases |

## License

[MIT](LICENSE)
