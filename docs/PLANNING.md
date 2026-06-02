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
| Candidate Oracle | ✅ 完成 | SFT+DPO 候选池上界 80.95% |
| Learned Selector | 🔄 探索中 | 有空间但未完整验证 |
| 跨数据集迁移 | 暂缓 | 需在目标数据集重新走完整 SFT → DPO 流程 |

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

### 跨数据集迁移暂缓

早期尝试直接用 Spider SFT 模型迁移到更复杂数据集时效果不理想。分析原因：

| 问题 | 说明 |
|------|------|
| **Schema 差异大** | 不同数据集的 database/schema 规模、列名风格和外键结构差异明显 |
| **领域偏移** | 学术场景训练数据难以覆盖金融、医疗等专业领域 |
| **SQL 复杂度** | 更复杂数据集的查询更长、嵌套更深，Spider 训练数据覆盖不足 |
| **评估标准不同** | 不同 benchmark 的执行标准和数据分布存在差异 |

**教训**：NL2SQL 的泛化极度依赖训练数据覆盖。Spider pipeline 验证的是 post-training 链路设计合理性，而不是训练出一个直接跨 benchmark 泛化的通用模型。迁移到新数据集时，应在目标数据集上重新走完整 SFT → SC → DPO 流程。

### DPO 数据质量教训

首轮 DPO 训练几乎无提升（71.95% vs 71.76%），事后排查发现：
- sampled-correct chosen 未充分门控，存在偶然正确风险
- 103 对 chosen/rejected 完全一致

**教训**：训练指标不反映数据质量，数据必须直接 inspect。

### Candidate Oracle 分析

为分析提升空间，计算了多 policy 候选池的 oracle（每道题都能从候选池中选到正确 SQL 的理论上界）：

| 候选池 / Selector | Spider Dev (EX) | 说明 |
|-------------------|:---------------:|------|
| DPO + SC-12 | 75.0% | 当前实际选择结果 |
| SFT + DPO Candidate Oracle | **80.95%** | 多 policy 候选池分析上界 |

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
- [x] local hard-negative 策略（gold chosen 为主，门控 sampled-correct 为辅）
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

Candidate Oracle 80.95% vs 当前 75.0%，有约 6% 的 gap 来自选择环节。

可选方向：
- **轻量 learned selector**：用小模型对候选 SQL 打分（对训练数据要求高）
- **Execution-aware verifier/reranker**：结合执行结果、schema 合法性和问题-SQL 对齐进行选择

### Phase 2：SFT/DPO 融合

DPO 在 extra 难度提升明显，但仍有少量 SFT-to-DPO regression。探索更深层的融合策略：
- conservative DPO / FTX 正则，保留 SFT 的 schema-grounded 稳定性
- anti-regression DPO，用训练集上的 SFT 正确、DPO 错误案例构造修复偏好
- 将 SFT 作为 reference/anchor，而不是简单做推理时双模型 ensemble

### Phase 3：新数据集完整复现

将 Spider pipeline 迁移到新数据集时，需**从头训练完整流程**：

```
目标数据集
  → SFT
  → SC 消融
  → DPO
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
