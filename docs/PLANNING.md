# AlignSQL 项目规划

> 当前状态 → 已完成工作 → 后续计划

---

## 一、当前状态

| 维度 | 状态 | 备注 |
|------|------|------|
| Spider SFT | ✅ 完成 | EX 71.8%, EM 67.0% |
| Spider SC | ✅ 完成 | SC-12: EX 74.2%, EM 69.0% |
| Spider DPO | ✅ 完成 | Greedy EX 74.0%, SC-12: EX 75.0% |
| Error Analysis | ✅ 完成 | 错误类型分类、Oracle 分析 |
| **BIRD 泛化** | **❌ 失败** | **直接使用 Spider 模型跑 BIRD 泛化极差** |
| Learned Selector | 🔄 探索中 | Candidate Oracle 80.6%, 有空间但未完整验证 |

### 各阶段实验结果

| 阶段 | Spider Dev (EX) | Exact Match |
|------|:---------------:|:-----------:|
| Zero-shot (baseline) | 43.9% | 35.7% |
| SFT (Greedy) | 71.8% | 67.0% |
| SFT + SC-12 | 74.2% | 69.0% |
| DPO (Greedy) | 74.0% | 68.1% |
| DPO + SC-12 | **75.0%** | 68.9% |

---

## 二、关键发现

### BIRD 泛化失败

尝试直接用 Spider SFT 模型跑 BIRD dev，结果远低于预期。分析原因：

| 问题 | 说明 |
|------|------|
| **Schema 差异大** | BIRD 的 database/schema 规模远超 Spider，最大单表数十列，prompt 截断严重 |
| **领域偏移** | 金融、医疗等专业领域，Spider 的学术场景数据无法覆盖 |
| **SQL 复杂度** | BIRD 的查询更长、嵌套更深，Spider 训练数据中罕见此类模式 |
| **评估标准不同** | BIRD 使用 VES（Value Exactness），与 Spider 的 EX/EM 不同 |

**教训**：NL2SQL 的泛化极度依赖训练数据的覆盖。Spider 上训练的能力不能直接迁移到 BIRD。正确的做法是：**在 BIRD 上走完全同样的 SFT → DPO 流程**，Spider pipeline 验证的是链路设计合理性，而不是训练出一个通用模型。

### DPO 数据质量教训

首轮 DPO 训练几乎无提升（71.95% vs 71.76%），事后排查发现：
- 35% 的 preference pair 中 `chosen != gold_sql`
- 103 对 chosen/rejected 完全一致

**教训**：训练指标不反映数据质量，数据必须直接 inspect。

### Candidate Oracle 分析

为分析提升空间，计算了多 policy 候选池的 oracle（每道题都能从候选池中选到正确 SQL 的理论上界）：

| 候选池 / Selector | Spider Dev (EX) | 说明 |
|-------------------|:---------------:|------|
| DPO + SC-12 | 75.0% | 当前实际选择结果 |
| SFT + DPO Candidate Oracle | **80.6%** | 多 policy 候选池分析上界 |

Oracle 结果说明正确 SQL 已经出现在候选池中，瓶颈正从「候选生成」转向「候选选择」。这是后续最有希望的方向。

---

## 三、当前流程

```
Spider 训练数据 (train_spider.json)
    ↓
SFT 训练 (LoRA)
    ↓
模型采样候选 SQL (多温度, T=0.8~1.6)
    ↓
执行对比 + 错误分类 → 构造 DPO pairs
    ↓
DPO 训练
    ↓
评测 (Greedy + SC-N)
    ↓
错误分析 + Oracle 分析
```

### 完成的工作

**SFT 链路**：
- [x] 数据预处理（Spider → Alpaca 格式）
- [x] Schema 序列化与 prompt 构建
- [x] LoRA 微调训练配置
- [x] 模型合并与推理评测

**Self-Consistency**：
- [x] 批量采样 N 个候选 + 执行投票
- [x] greedy / SC-N 评测脚本
- [x] N=5/8/12 消融实验

**DPO 链路**：
- [x] 候选采样（多温度、多轮次）
- [x] SQL 质量过滤（防噪声污染）
- [x] 执行对比 + 错误分类（13 类本地错误）
- [x] local hard-negative 策略（chosen = gold SQL）
- [x] 变异负样本（规则生成）
- [x] 错误类型平衡
- [x] LLaMA-Factory DPO 训练配置
- [x] 整个 pipeline 跑通并记录结果

**实验记录**：
- [x] 实验结果表格（README）
- [x] SFT / DPO 训练文档
- [x] Candidate Oracle 分析

---

## 四、后续计划

### Phase 1：候选选择（最优先）

Candidate Oracle 80.6% vs 当前 75.0%，有 ~5.6% 的 gap 来自选择环节。

可选方向：
- **规则 selector**：基于执行结果的启发式选择（已初步实验，75.8%）
- **轻量 learned selector**：用小模型对候选 SQL 打分（对训练数据要求高）
- **Execution-aware reranker**：边执行边剪枝

### Phase 2：SFT/DPO 融合

DPO 在 extra 难度提升 +5.4%，但在 easy 略有下降。探索融合策略：
- 多 checkpoint 集成
- SFT + DPO 联合解码

### Phase 3：新数据集完整复现

将 Spider pipeline 迁移到新数据集（如 BIRD）时，需**从头训练完整流程**：

```
BIRD 数据
  → BIRD SFT （不是用 Spider 模型直接跑）
  → BIRD SC 消融
  → BIRD DPO
  → 评估
```

预期在迁移过程中会发现 pipeline 的不足，反过来改进设计。

---

## 五、项目结构

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
│   ├── analyze_difficulty.py
│   ├── run_zeroshot.sh
│   └── run_eval.sh                # 评测入口
├── docs/
├── outputs/                       # 实验结果
├── dataset/                       # 原始数据
└── data_processed/                # 预处理产物
```

---

## 六、后续预期

| 方向 | 预期收益 | 不确定性 |
|------|---------|---------|
| Candidate Selection | +3~5% EX | Learned selector 需要训练数据 |
| SFT/DPO 融合 | +1~2% EX | 收益可能不大 |
| 新数据集全流程 | 验证 pipeline 可迁移性 | 取决于数据集规模和难度 |
