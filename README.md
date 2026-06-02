# AlignSQL：NL2SQL 全流程 Post-training Pipeline

> 从 SFT 到 DPO 再到 RL，完整跑通 NL2SQL 的模型对齐实践。
>
> 基座模型：Qwen3-8B | 框架：LLaMA-Factory | 硬件：RTX 4090 (24GB)

---

## 实验结果

| 阶段 | Spider Dev (EX) | Exact Match | 状态 |
|------|:---------------:|:-----------:|:----:|
| Zero-shot (baseline) | 43.91% | 35.69% | ✅ |
| SFT (Greedy) | 71.76% | 67.02% | ✅ |
| SFT + SC-5 | 73.02% | 68.38% | ✅ |
| SFT + SC-8 | **74.27%** | 68.57% | ✅ |
| SFT + SC-12 | 74.18% | **68.96%** | ✅ |

### 按难度分级

详见 [Self-Consistency 文档](docs/self-consistency.md)。

## 快速开始

> ⚠️ **注意**：以下脚本中的模型路径（如 `/root/autodl-tmp/models/...`）为参考路径，请根据实际情况修改。

```bash
# Zero-shot 评测（基座模型）
bash scripts/run_zeroshot.sh

# SFT 训练与评测
python scripts/prepare_sft.py
llamafactory-cli train configs/spider/sft.yaml
bash scripts/run_eval.sh

# Self-Consistency 消融（N=5,8,12）
bash scripts/run_eval.sh 5   # SC N=5
bash scripts/run_eval.sh 8   # SC N=8
bash scripts/run_eval.sh 12  # SC N=12
```

### 自定义模型路径

```bash
# 通过 --model_path 指定
bash scripts/run_eval.sh --model_path /your/path/to/model 5
```

## 经验教训

> 以下问题均源于真实踩坑，记录以供后续参考。

### DPO 数据质量是首要问题

首轮 DPO 训练后，Spider EX 仅 71.95%（SFT baseline 71.76%），几乎没有提升。事后分析发现：
- **35% 的 preference pair 中 `chosen != gold_sql`** ——模型在偏好错误的 SQL，而非 gold SQL。根因是 `prepare_dpo.py` 将 SFT 正确预测也加入了 chosen，这些预测在 EX 上正确但可能与 gold SQL 字面不同。
- **103 对 chosen / rejected 完全一致** ——模型从中学不到任何偏好信号。

教训：DPO 训练前必须检查数据质量，至少验证 `chosen==gold_sql` 的比例和 chosen/rejected 的 divergence。

### 优先用干净数据验证方法可行性

不应直接投入完整（可能含噪的）数据训练。先过滤出 gold-only pairs 跑一轮小实验，确认 DPO 在理想数据上有效，再逐步放开数据范围。

### 训练指标不能替代数据检查

首轮 DPO 训练中 `reward_accuracy` 等指标表现正常，并未提示数据有问题。但事后检查才发现 35% 的 pair `chosen != gold_sql`。**训练指标不反映「偏好是否正确」，只反映「偏好是否被优化」**——数据质量必须直接 inspect，不能靠汇总指标间接判断。

### Prompt 一致性

DPO 训练和评测使用不同 prompt 格式会导致分数虚低。虽然本次测试证明 prompt 不一致不是主因（SFT prompt 下 DPO 仍无提升），但仍需作为常规检查项。

## 项目结构

```
├── alignsql/                      # Python 包 (pip install -e .)
│   ├── __init__.py
│   ├── data/                      # 数据加载/处理
│   │   ├── preprocessing.py       # 难度分类、Prompt 构建
│   │   ├── schema.py              # Schema 序列化
│   │   └── spider.py              # Spider 数据加载器
│   ├── models/                    # 模型推理策略
│   │   └── inference.py           # Self-Consistency 采样 & 投票
│   ├── eval/                      # 评估指标
│   │   └── metrics.py
│   ├── analysis/                  # 错误分析 (待实现)
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
│   ├── evaluate_vllm.py           # 推理评测 (greedy / SC)
│   ├── prepare_sft.py
│   ├── analyze_difficulty.py
│   ├── run_zeroshot.sh
│   ├── run_sft.sh
│   └── run_sc.sh                  # SC 消融
├── tests/
├── outputs/                       # 实验结果
├── assets/                        # 实验图表
├── docs/
├── dataset/                       # 原始数据 (Spider JSON + SQLite)
├── data_processed/                # 预处理产物
├── setup.py
├── Makefile
├── LICENSE
└── .gitignore
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
