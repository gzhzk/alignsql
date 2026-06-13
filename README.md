# AlignSQL：NL2SQL 全流程 Post-training Pipeline

> 从 SFT 到 DPO 再到 RL，完整跑通 NL2SQL 的模型对齐实践。
>
> 基座模型：Qwen3-8B | 框架：LLaMA-Factory | 硬件：RTX 4090 (24GB)

---

## 实验结果

| 阶段 | Spider Dev (EX) | Exact Match |
|------|:---------------:|:-----------:|
| Zero-shot (baseline) | 43.91% | 35.69% |
| SFT (Greedy) | 71.76% | 67.02% |
| SFT + SC-5 | 73.02% | 68.38% |
| SFT + SC-8 | 74.27% | 68.57% |
| SFT + SC-12 | 74.18% | **68.96%** |
| DPO (Greedy) | **73.98%** | 68.09% |
| DPO + SC-5 | 74.56% | 68.18% |
| DPO + SC-8 | 74.47% | 67.99% |
| DPO + SC-12 | **74.95%** | 68.86% |

---

## Pipeline 一览

### Zero-shot → SFT：从基座模型到可用

基座 Qwen3-8B 在 Spider 上只有 43.9% EX。SFT 用 7K 训练数据 + LoRA 微调后达到 **71.8% EX**，解决了大部分基础问题（easy: 89%, medium: 74%），但 hard/extra 仍然偏低。

### Self-Consistency：不训练也能提点

对每条问题采样 N 个候选，执行后 majority vote。SC-8 达到 **74.3% EX**，比 greedy 高 2.5%。这意味着模型本身已经知道正确答案，只是单次生成不稳定。

### DPO：从选对到偏对

用 SFT 模型采样的候选构造 preference pairs，让模型学习偏好正确的 SQL 生成行为：

1. 对每条训练问题采样候选 SQL，执行对比 gold 结果
2. chosen 以 gold SQL 为主，rejected 用局部错误分类自动打标签
3. 按错误类型平衡数据，避免只学到避免某一种错误

DPO 带来 **+2.2% EX**（greedy），收益集中在 medium/extra 上（extra +5.4%）。逐题对比显示：DPO 修复 51 个 SFT 错题，引入 28 个回归，净修复 23 题。

### Candidate Oracle：瓶颈在哪

如果每道题都能从候选池中选到正确 SQL，理论上能达到多少？

| 候选池 / Selector | Spider Dev (EX) | 说明 |
|-------------------|:---------------:|------|
| DPO + SC-12 | 75.0% | 当前实际结果 |
| SFT + DPO Candidate Oracle | **81.43%** | 多 policy 候选池上界 |

正确答案已经在候选池中了。瓶颈从**候选生成**转向了**候选选择**。

### 一条主线

```
SFT (71.8%) ── 决定能力基线
   ↓
SC (74.3%) ── 暴露不稳定：模型知道但选不对
   ↓
DPO (75.0%) ── 偏好修正稳步提点，但收益递减
   ↓
Oracle (81.43%) ── 瓶颈在"选择"不在"生成"
   ↓
RL ── 待探索
```

---

## 快速开始

> ⚠️ 以下路径为参考，请根据实际情况修改。

```bash
# Zero-shot 评测（基座模型）
bash scripts/run_zeroshot.sh

# SFT 训练与评测
python scripts/prepare_sft.py
llamafactory-cli train configs/spider/sft.yaml
bash scripts/run_eval.sh                               # greedy + SC-5/8/12

# DPO 数据准备
python scripts/prepare_dpo.py \
    --model_path /root/autodl-tmp/models/sft/merged \
    --spider_dir dataset/spider \
    --output data_processed/dpo_pairs.json

# DPO 训练
llamafactory-cli train configs/spider/dpo.yaml

# DPO 评测
bash scripts/run_eval.sh --model dpo                    # 全量消融
bash scripts/run_eval.sh --model dpo --greedy           # 仅 greedy
bash scripts/run_eval.sh --model dpo 5                  # 仅 SC-5
bash scripts/run_eval.sh --model /custom/path 12        # 自定义路径
```

---

## 项目结构

```
alignsql/
├── alignsql/                      # Python 包 (pip install -e .)
│   ├── data/                      # 数据加载/处理
│   │   ├── preprocessing.py       # 难度分类、Prompt 构建
│   │   ├── schema.py              # Schema 序列化
│   │   └── spider.py              # Spider 数据加载器
│   ├── models/
│   │   └── inference.py           # Self-Consistency 采样 & 投票
│   ├── eval/
│   │   └── metrics.py
│   └── utils/
│       ├── db.py                  # SQLite 执行工具
│       └── io.py                  # JSON/JSONL 读写
├── vendor/                        # 第三方代码 (Spider 官方评测)
├── configs/
│   └── spider/
│       ├── sft.yaml
│       └── merge_sft.yaml
├── scripts/
│   ├── evaluate_vllm.py           # 推理评测 (greedy / SC)
│   ├── prepare_sft.py             # SFT 数据准备
│   ├── prepare_dpo.py             # DPO 数据准备
│   ├── oracle_analysis.py         # 候选池 oracle 分析
│   ├── analyze_difficulty.py
│   ├── run_zeroshot.sh
│   └── run_eval.sh                # 评测入口
├── docs/
├── outputs/                       # 实验结果
├── dataset/                       # 原始数据
└── data_processed/                # 预处理产物
```

---

## 深入分析

### DPO 提升分析

DPO 的主要收益集中在 medium 以上难度：

| 难度 | SFT EX | DPO EX | 提升 |
|------|:------:|:------:|:----:|
| easy | 89.1% | 88.7% | -0.4% |
| medium | 74.4% | 76.0% | +1.6% |
| hard | 65.5% | 67.2% | +1.7% |
| extra | 48.2% | 53.6% | **+5.4%** |
| **all** | **71.8%** | **74.0%** | **+2.2%** |

逐题对比：DPO 修复 51 个 SFT 错题，引入 28 个回归，净修复 23 题。修复主要集中在 WHERE 条件、表选择和复杂过滤相关错误。

### Candidate Oracle

为定位后续提升空间，计算了多 policy 候选池的 oracle——如果每道题都能选到正确候选的理论上界：

```bash
# SFT + DPO multi-policy candidate oracle
python scripts/oracle_analysis.py \
    --spider_dir dataset/spider \
    --candidate_file outputs/spider/sft/sc_n12/candidates.json \
    --candidate_file outputs/spider/dpo/sc_n12/candidates.json \
    --reference_results outputs/spider/dpo/sc_n12/results.json
```

Oracle 81.43% 说明模型能生成正确答案，但当前 majority vote 选不出。转向 execution-aware verifier 或 learned reranker 是比较明确的后续方向。

### Post-training 思考

本项目的实验也暴露出一个更一般的问题：SFT、DPO、SC 不是彼此孤立的技巧，而是围绕同一个目标的不同环节。

- **SFT 提供 schema-grounded base policy**——格式稳定、列名和表名 grounding 较好，是后续对齐的锚点
- **DPO 提供 preference shift**——把模型从局部错误中拉出来，但如果约束不足，也可能带来少量 schema drift
- **SC 暴露候选池上限**——Oracle 分析达到 80%+ 说明瓶颈不完全是生成能力，而是如何选择候选
- **候选选择应从 vote 走向 verifier**——简单 majority vote 无法识别"少数但正确"的 SQL
- **SFT/DPO 的深层融合比简单 ensemble 更重要**——通过 conservative DPO、FTX 正则或 anti-regression DPO，保留 SFT 的稳定性同时吸收 DPO 的偏好修正
- **RL 应建立在清晰瓶颈之上**——当选择器已接近上限、剩余问题确实来自生成能力时，再引入 execution reward 的 GRPO 更合理

后续路线：SFT 建立可用策略 → DPO 修正偏好 → Oracle 定位瓶颈 → Verifier 改善候选选择 → RL 将执行反馈回灌到单一 policy 中。

---

## 经验教训

### DPO 数据质量是首要问题

首轮 DPO 训练后 EX 仅 71.95%（baseline 71.76%），几乎无效。事后排查发现：
- 部分 sampled-correct chosen 只是偶然执行正确，与 gold SQL 结构差异大
- 103 对 chosen/rejected 完全一致

教训：`reward_accuracy` 正常不代表数据没问题。训练前必须 inspect chosen 来源分布和 chosen/rejected 的 divergence。

### 优先用干净数据验证方法可行性

先过滤出 gold-only pairs 跑一轮小实验，确认 DPO 在理想数据上有效，再逐步放开数据范围。

### 跨数据集迁移需重新训练

早期尝试将 Spider 模型直接迁移到其他 NL2SQL 数据集，效果不理想。原因是 NL2SQL 的泛化依赖训练数据覆盖。Spider pipeline 验证的是 post-training 链路设计，而不是训练通用模型。迁移到新数据集时需重新走完整 SFT → DPO 流程。

---

## 详细方案

- [项目规划](docs/PLANNING.md)
- [Oracle 归因分析](docs/oracle-attribution.md)
- [SFT 训练流程](docs/sft.md)
- [DPO 偏好对齐](docs/dpo.md)

## 技术栈

| 组件 | 选型 |
|------|------|
| 基座模型 | Qwen3-8B |
| 微调框架 | LLaMA-Factory (LoRA) |
| 推理加速 | vLLM |
| 实验追踪 | Weights & Biases |
