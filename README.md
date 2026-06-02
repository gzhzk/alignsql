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
| SFT + SC-8 | 74.27% | 68.57% | ✅ |
| SFT + SC-12 | 74.18% | **68.96%** | ✅ |
| DPO (Greedy) | **73.98%** | 68.09% | ✅ |
| DPO + SC-5 | 74.56% | 68.18% | ✅ |
| DPO + SC-8 | 74.47% | 67.99% | ✅ |
| DPO + SC-12 | **74.95%** | 68.86% | ✅ |

DPO 训练相比 SFT baseline 提升约 **+2.2% EX**（greedy），结合 SC-12 达到最佳 **75.0% EX**。详见 [Self-Consistency 文档](docs/self-consistency.md)。

**关键发现**

- DPO 的收益主要来自 medium / extra-hard SQL，而不是 easy case。
- Greedy 下 DPO 修复 51 个 SFT 错题，同时引入 28 个回归，净修复 23 题。
- Candidate oracle 显示，SFT+DPO 候选池存在 **80.95% EX** 的分析上界；当前瓶颈开始从候选生成转向候选选择。
- 后续方向是 SFT/DPO 深层融合与 execution-aware verifier，而不是简单继续加长训练。

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

# DPO 数据准备
python scripts/prepare_dpo.py \
    --model_path /root/autodl-tmp/models/sft/merged \
    --spider_dir dataset/spider \
    --output data_processed/dpo_pairs.json

# DPO 训练
llamafactory-cli train configs/spider/dpo.yaml

# DPO 评测
bash scripts/run_eval.sh --model dpo
bash scripts/run_eval.sh --model dpo 5
bash scripts/run_eval.sh --model dpo 8
bash scripts/run_eval.sh --model dpo 12
```

### 自定义模型路径

```bash
# 通过 --model_path 指定；--model 决定 prompt/stage
bash scripts/run_eval.sh --model dpo --model_path /your/path/to/model 5
```

## 实验分析

### Final DPO 设置

最终 DPO 实验使用清洗后的 preference pairs，并转换为 LLaMA-Factory DPO JSONL 格式：

| 项目 | 设置 |
|------|------|
| 数据文件 | `data_processed/dpo_data_clean_n16.jsonl` |
| 样本数 | 5,443 pairs（由 5,455 clean pairs 去除 12 个 identical pairs 后得到） |
| Chosen 来源 | `gold`: 3,360；`sampled_correct`: 2,095 |
| 训练模型 | `/root/autodl-tmp/models/sft/merged` |
| 训练轮数 | 2 epochs |
| 学习率 | `1e-5` |
| DPO beta | `pref_beta=0.1` |
| FTX 正则 | `pref_ftx=0` |
| Prompt 格式 | SFT prompt / `qwen3_nothink` |

### DPO 提升分析与 Candidate Oracle

DPO 的主要收益并非来自 easy case，而是集中在更复杂的 SQL 上。Greedy 评测中，DPO 相比 SFT 在 `extra` 难度上从 **46.39% EX** 提升到 **53.61% EX**（+7.23），`medium` 从 **73.77% EX** 提升到 **76.01% EX**（+2.24）。逐题对比显示，DPO 修复了 51 个 SFT 错题，同时引入 28 个回归，净修复 23 题；修复主要集中在 `WHERE` 条件、表选择和复杂过滤相关错误。

为分析后续提升空间，我们进一步计算了候选池 oracle。该指标表示：如果每道题都能从候选 SQL 中选到执行正确的那一个，理论上可达到的 EX 上界。它不是实际部署分数，也不作为主结果，只用于定位瓶颈。

| 候选池 / Selector | Spider Dev (EX) | 说明 |
|-------------------|:---------------:|------|
| DPO + SC-12 | 74.95% | 当前实际选择结果 |
| DPO-only Candidate Oracle | 78.34% | 单 DPO policy 候选池上界 |
| SFT + DPO Candidate Oracle | **80.95%** | 多 policy 候选池分析上界，非实际推理分数 |

Oracle 结果说明：当前模型并非完全生成不出正确 SQL，正确答案已经较多出现在候选池中；瓶颈开始从「候选生成」转向「候选选择」。受限于时间，本项目尚未完整验证 learned selector。后续计划探索更统一的 **SFT/DPO 深层融合** 与 **execution-aware reranker/verifier**：利用 SFT 的 schema-grounded 稳定性和 DPO 的偏好修正能力，减少 DPO 带来的 schema drift，同时从候选池中更可靠地选择正确 SQL。

Oracle 复现命令：

```bash
# DPO-only candidate oracle
python scripts/oracle_analysis.py \
    --spider_dir dataset/spider \
    --candidate_file outputs/spider/dpo/sc_n12/candidates.json \
    --reference_results outputs/spider/dpo/sc_n12/results.json

# SFT + DPO multi-policy candidate oracle
python scripts/oracle_analysis.py \
    --spider_dir dataset/spider \
    --candidate_file outputs/spider/sft/sc_n12/candidates.json \
    --candidate_file outputs/spider/dpo/sc_n12/candidates.json \
    --reference_results outputs/spider/dpo/sc_n12/results.json
```

### Post-training 思考

本项目的实验也暴露出一个更一般的 post-training 问题：SFT、DPO、SC 和后续 RL 并不是彼此孤立的技巧，而是围绕同一个目标的不同环节。

- **SFT 提供 schema-grounded base policy**：SFT 的优势是格式稳定、列名和表名 grounding 较好，是后续对齐的锚点。
- **DPO 提供 preference shift**：DPO 能把模型从容易犯的局部错误中拉出来，尤其改善 `WHERE`、表选择和 extra-hard SQL，但如果约束不足，也可能带来少量 schema drift 或 SFT-to-DPO regression。
- **SC 暴露候选池上限**：SC 不只是提升分数的推理技巧，也是一种诊断工具。Candidate oracle 达到 80%+ 说明模型已经能生成不少正确 SQL，后续瓶颈不完全是生成能力，而是如何选择候选。
- **候选选择应从 vote 走向 verifier**：简单 majority vote 太粗糙，无法识别“少数但正确”的 SQL。更合理的方向是 execution-aware verifier / learned reranker，让选择器理解 SQL 结构、schema 合法性和问题-SQL 对齐。
- **SFT/DPO 的深层融合比简单 ensemble 更重要**：直接同时跑 SFT 和 DPO 可以作为 oracle 分析，但推理成本和方法叙事都不够干净。更统一的方向是把 SFT 作为 reference/anchor，通过 conservative DPO、FTX 正则或 anti-regression DPO，保留 SFT 的稳定性，同时吸收 DPO 的偏好修正能力。
- **GRPO/RL 应建立在清晰瓶颈之上**：当候选池分析表明选择器已接近上限、剩余问题确实来自生成能力时，再引入 execution reward 的 GRPO 更合理。否则，盲目上 RL 可能只是放大已有 selector 或数据噪声。

因此，后续路线不是简单地“继续训更久”，而是逐步形成一个更闭环的 post-training pipeline：SFT 建立可用策略，DPO 修正偏好，SC/Oracle 定位瓶颈，Verifier 改善候选选择，最后再用 RL 将执行反馈进一步回灌到单一 policy 中。

## 经验教训

> 以下问题均源于真实踩坑，记录以供后续参考。

### DPO 数据质量是首要问题

首轮 DPO 训练后，Spider EX 仅 71.95%（SFT baseline 71.76%），几乎没有提升。事后分析发现：
- **未经门控的 `sampled_correct` chosen 风险较高** ——执行正确但非 gold 的 SQL 可能只是当前数据库状态下偶然正确，若不做相似度、结构和结果质量门控，会把不稳定 SQL 当成偏好目标。
- **103 对 chosen / rejected 完全一致** ——模型从中学不到任何偏好信号。

教训：DPO 训练前必须检查数据质量，至少验证 chosen 来源分布、chosen/rejected 的 divergence、SQL 文本噪声，以及 sampled-correct chosen 是否通过质量门控。`chosen != gold_sql` 并非一定不可用，但必须谨慎过滤。

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
│   ├── prepare_sft.py             # SFT 数据准备
│   ├── prepare_dpo.py             # DPO 数据准备
│   ├── oracle_analysis.py         # 候选池 oracle 上界分析
│   ├── analyze_difficulty.py
│   ├── run_zeroshot.sh
│   └── run_eval.sh                # 评测入口（greedy / SC）
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
- [DPO 偏好对齐](docs/dpo.md)
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
