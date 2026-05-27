# AlignSQL 重构计划

> 当前状态评估 → 目标架构 → 分阶段实施路线

---

## 一、当前状态评估

| 维度 | 现状 | 问题 |
|------|------|------|
| 数据集 | 仅 Spider（7K 样本，小规模学术场景） | 说服力弱，BIRD 更接近真实场景 |
| 训练流程 | SFT 完成（72.24%），DPO 未跑 | 核心差异化部分缺失 |
| 推理策略 | 贪心解码（temperature=0），单次生成 | 无自纠错、无候选投票 |
| 代码结构 | 平铺脚本，无包结构 | 不可 import，无测试，工程感弱 |
| 实验管理 | 有 wandb 配置，无 ablation 对比 | 无超参消融、无错误分析 |
| 文档 | 详细但偏"方案设想"，缺少实际经验总结 | 项目报告像实验计划书而非实验报告 |

---

## 二、目标架构

```
┌──────────────────────────────────────────────────┐
│                   数据集层                         │
│  Spider (playground, 快速迭代)                     │
│  BIRD (泛化验证, 真实场景)                         │
│  Schema Filtering (BIRD schema 过长时启用)         │
└───────────────┬──────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────┐
│                  训练层                            │
│  Phase 1: SFT (LoRA)                              │
│  Phase 2: DPO (执行反馈自动构建偏好)               │
│  Phase 3: GRPO (以 EX 为 reward 强化学习)         │
│  [可选] Iterative DPO (多轮迭代)                   │
└───────────────┬──────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────┐
│                  推理层                            │
│  Self-Consistency (采样 N 个 → 执行 → 投票)       │
│  Execution-guided Regeneration (报错重生成)        │
└───────────────┬──────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────┐
│                  评估层                            │
│  EX / EM 分难度评估                               │
│  Error Analysis (错误类型分类 + 可视化)            │
│  Ablation 实验对比                                │
└──────────────────────────────────────────────────┘
```

---

## 三、项目结构（当前状态）

```
AlignSQL/
├── alignsql/                      # Python 包 (pip install -e .)
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py       # 难度分类、Prompt 构建
│   │   ├── schema.py              # Schema 序列化
│   │   └── spider.py              # Spider 数据加载器
│   ├── models/
│   │   ├── __init__.py
│   │   └── inference.py           # Self-Consistency 采样 & 投票
│   ├── eval/
│   │   ├── __init__.py
│   │   └── metrics.py             # 结果比较、分数统计
│   ├── analysis/
│   │   └── __init__.py            # 错误分析 (待实现)
│   └── utils/
│       ├── __init__.py
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
├── scripts/                       # 可执行入口 (薄层调用)
│   ├── evaluate_vllm.py           # 推理评测 (greedy / SC 两模式)
│   ├── prepare_sft.py             # SFT 数据预处理
│   ├── analyze_difficulty.py      # 难度分析
│   ├── run_zeroshot.sh
│   ├── run_sft.sh
│   └── run_sc.sh                  # Self-Consistency 消融
├── tests/
│   ├── test_data.py
│   ├── test_schema.py
│   └── test_utils.py
├── outputs/                       # 实验结果
│   ├── zeroshot/results.json
│   └── sft/results.json
├── assets/                        # 实验图表
├── docs/
│   └── PLANNING.md               # 本文件
├── data_processed/                # 预处理产物 (sft_data.json)
├── dataset/                       # 原始数据 (Spider JSON + SQLite)
├── setup.py
├── Makefile
├── .gitignore
└── README.md
```

### 项目结构（后续按需添加）

```
configs/
├── bird/                   # BIRD 训练配置
│
alignsql/
├── data/
│   └── bird.py             # BIRD 数据加载器
└── models/
    └── dpo_trainer.py      # DPO 训练逻辑
```

---

## 四、分阶段实施路线

### Phase 0：代码重构（1 天）✅

目标：从"脚本集合"变为"可安装包"

- [x] 创建 `alignsql/` 包结构（data / models / eval / analysis / utils）
- [x] 抽取公共逻辑（db 工具、schema 序列化、数据加载）
- [x] 迁移第三方代码到 `vendor/`（evaluation.py + process_sql.py）
- [x] 编写 `setup.py` + `Makefile`
- [x] 迁移现有实验数据到 `outputs/` 目录
- [x] 将 Self-Consistency 逻辑封装到 `alignsql/models/inference.py`
- [x] 集成 SC 到 `evaluate_vllm.py`（`--self_consistency` 开关）
- [x] 补基础测试（data / schema / utils）
- [x] 配置迁移：`config/` → `configs/spider/`

### Phase 1：Self-Consistency 推理（1 天）✅

目标：不改模型，最快提点

- [x] 实现 `alignsql/models/inference.py` — 批量采样 N 个候选 + 执行投票
- [x] 集成到 `evaluate_vllm.py`（`--self_consistency` 开关 + `--n_candidates`）
- [x] 提供 `scripts/run_sc.sh` 消融脚本（greedy + N=3/5/8）
- [ ] 对比不同 N（3/5/8/16）的效果（需 GPU 运行）
- [ ] 在 Spider Dev 上跑通并记录结果
- [ ] 移植到 BIRD 验证

预计提升：+3~8% EX（Spider 上）

### Phase 2：DPO 完整跑通（2-3 天）

目标：SFT → DPO 完整链路

- [ ] 实现 `generate_candidates.py` — 候选 SQL 生成（beam + sampling）
- [ ] 实现 `build_preferences.py` — 执行反馈构建偏好对
- [ ] 写 `configs/spider/dpo.yaml`
- [ ] 训练 + 评估
- [ ] 迭代 DPO 第二轮

### Phase 3：BIRD 接入（2-3 天）

目标：在更难的数据集上验证泛化能力

- [ ] 下载 BIRD 数据集
- [ ] 实现 `alignsql/data/bird.py` — BIRD 数据加载与预处理
- [ ] 实现 Schema Filtering（BIRD schema 长，需选择相关表）
- [ ] 在 BIRD 上跑 SFT + Self-Consistency
- [ ] 对比 Spider vs BIRD 的效果差异

### Phase 4：GRPO 强化学习（3-5 天，探索性）

目标：以 SQL 执行结果为 reward，用 RL 进一步对齐

- [ ] 调研 LLaMA-Factory 的 GRPO 支持（需 dpo 最新版）
- [ ] 设计 reward 函数（EX score + 执行效率 + 语法合法性）
- [ ] 编写 `configs/spider/grpo.yaml`
- [ ] 训练 + 评估
- [ ] 对比 SFT vs DPO vs GRPO

注意：此阶段效果不确定，GRPO 在数学推理上效果好但在 SQL 上缺乏验证。

### Phase 5：错误分析 + 消融实验（1-2 天）

目标：不仅报告结果，更分析原因

- [ ] 实现 `eval/error_analysis.py` — 按错误类型分类
- [ ] 分析 SFT 模型的典型错误模式
- [ ] 做 ablation：
  - LoRA rank 消融（rank=4/8/16/32）
  - Learning rate 消融（1e-4/2e-4/5e-4）
  - Schema 模板消融（不同 prompt 风格）
- [ ] 输出实验报告

---

## 五、关键设计决策

### Schema Filtering（BIRD）

BIRD 数据库大（最大单库 36GB），schema 可能包含数百列。直接全量塞入 prompt 会超长。

方案：基于 question 和 schema 元素的语义相似度做筛选。
- 对 question 做简单 tokenize，提取关键词
- 对每个表和列计算与 question 的 token 重叠
- 选择 Top-K 表（K 根据 context window 动态调整）
- 保留外键信息以支持跨表查询

### Self-Consistency 投票机制

```
对每个 question:
  1. 采样 N 个候选 SQL (temperature=0.8, top_p=0.9)
  2. 对每个候选执行并获取结果集
  3. 按结果集去重，选择出现次数最多的结果集对应的 SQL
  4. 平局时优先选执行时间短的

与 beam search 的区别：beam search 生成 top-K 概率路径，多样性不足。
采样能产生更多样化的候选，更适合投票。
```

### 实验命名规范

```
{stage}_{dataset}_{关键参数}_run{序号}

sft_spider_lr2e4_rank8_run001
sft_spider_lr2e4_rank16_run001
dpo_spider_beta0.3_run001
selfcons_spider_n8_run001
grpo_spider_lr1e5_run001
```

---

## 六、硬件与资源

| 组件 | 规格 | 预计开销 |
|------|------|----------|
| GPU | RTX 4090 (24GB) × 1 | 已有 |
| Spider SFT | ~2 小时 | 已有数据 |
| Spider DPO | ~1 小时 | 需运行 |
| BIRD SFT | ~4-6 小时 | 需运行 |
| BIRD DPO | ~2-3 小时 | 需运行 |
| Self-Consistency | 推理耗时 x N | 需运行 |
| 磁盘 | BIRD 约 33GB，Spider 约 2GB | 需下载 |

BIRD 数据较大，建议评估时对 BIRD 使用 subset（选择 5-10 个有代表性的数据库）来加速迭代。

---

## 七、预期成果

| 阶段 | Spider EX | BIRD EX | 说明 |
|------|-----------|---------|------|
| Zero-shot baseline | 43.91% | ~25-30% | 原始模型 |
| SFT | 72.24% | ~45-55% | LoRA 微调 |
| SFT + Self-Consistency | ~75-78% | ~50-58% | 推理时投票 |
| DPO | ~75-80% | ~50-60% | 偏好对齐 |
| DPO + Self-Consistency | ~78-82% | ~55-62% | 叠加效果 |
| GRPO | 待定 | 待定 | 探索性 |

BIRD 上的绝对数值低于 Spider 是正常的（BIRD 更难），有价值的是**相对于 baseline 的提升百分比**。
