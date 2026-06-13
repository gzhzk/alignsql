# Candidate Oracle 归因分析

> 目标：搞清楚 SFT+DPO 候选池 oracle 的收益来自哪里，以及后续 verifier / reranker 应该解决什么问题。

## 结论概览

当前实际最强 selector 是 DPO self-consistency：

| Selector | Spider Dev EX |
|----------|:-------------:|
| SFT SC-12 | 74.18% |
| DPO SC-12 | **74.95%** |
| SFT + DPO Candidate Oracle | **81.43%** |

因此，相比 DPO SC-12，候选池 oracle 还有约 **6.5 个点** 的空间。这个 gap 不是单纯的生成能力问题：很多题里正确 SQL 已经出现在候选池中，但多数投票 / 执行结果投票没有选出来。

最关键的归因结论是：

> 可追回错误的主因不是 tie-break，而是 **minority-correct candidates**：正确 SQL 存在，但票数少于某个可执行的错误结果簇。

换句话说，后续 verifier 不能只是“更会投票”，而要能识别少数但正确的候选。

## Oracle 重新计算

之前文档中记录的 oracle 是 **80.95%**。修正 oracle 评估中的 SQL identity 和 timeout cache 口径后，新的结果是：

```text
SFT + DPO Candidate Oracle: 842 / 1034 = 81.43%
```

按难度拆分：

| 难度 | Oracle Correct | Total | Oracle EX |
|------|---------------:|------:|----------:|
| easy | 228 | 248 | 91.94% |
| medium | 378 | 446 | 84.75% |
| hard | 131 | 174 | 75.29% |
| extra | 105 | 166 | 63.25% |
| all | 842 | 1034 | 81.43% |

## 之前 Oracle 错在哪里

旧 oracle 数字受两个评估细节影响。

第一，旧版 `normalize_sql()` 会把整条 SQL lower 掉，然后作为执行缓存 key：

```python
sql.strip().rstrip(";").lower()
```

这不安全。SQL 关键字和很多标识符在 SQLite 中通常不区分大小写，但字符串字面量不能随便 lower。

例如：

```sql
WHERE hometown != 'Little Lever Urban District'
```

和：

```sql
WHERE hometown != 'little lever urban district'
```

在 SQLite 默认语义下不一定等价。

这个 bug 的本质是把：

```text
SQL surface normalization
```

误用到了：

```text
execution semantics
```

第二，执行缓存 key 没有区分 `max_ops`。gold SQL 使用：

```text
gold_max_ops = 1_000_000
```

candidate SQL 原来默认使用：

```text
max_ops = 200_000
```

如果 gold 和 candidate SQL 文本相同，candidate 可能直接复用 gold 的执行结果，从而绕过 candidate 自己的执行步数限制。

修正后的缓存 identity 是：

```python
key = (db_id, normalize_sql(sql), max_ops)
```

同时 `normalize_sql()` 只规整空白和末尾分号，保留字符串 literal 的大小写。

## 题目级分桶

以 SFT SC-12 和 DPO SC-12 作为当前实际 selector，每道 Spider dev 题可以分成五类：

| Bucket | 含义 | 数量 |
|--------|------|-----:|
| A | SFT 对，DPO 对 | 733 |
| B | SFT 对，DPO 错 | 34 |
| C | SFT 错，DPO 对 | 42 |
| D | SFT 错，DPO 错，但 oracle 对 | 37 |
| E | SFT 错，DPO 错，oracle 也错 | 188 |

按难度拆分：

| Bucket | easy | medium | hard | extra |
|--------|-----:|-------:|-----:|------:|
| A | 219 | 324 | 114 | 76 |
| B | 4 | 19 | 3 | 8 |
| C | 1 | 22 | 7 | 12 |
| D | 5 | 16 | 7 | 9 |
| E | 19 | 65 | 43 | 61 |

这说明 verifier 问题可以拆成两个子问题：

1. **Policy conflict selection**：处理 B/C，即 SFT 和 DPO 一个对一个错的情况，共 **76** 题。
2. **True candidate reranking**：处理 D，即 SFT/DPO 两个 SC 输出都错，但合并候选池里有正确 SQL，共 **37** 题。

这两个问题不同。前者更像“什么时候信 SFT，什么时候信 DPO”；后者才是真正的候选重排。

## DPO 错但 Oracle 可修的空间

相对 DPO SC-12，有 **71** 道题 DPO 当前选错，但 SFT+DPO 合并候选池中存在正确 SQL。

按难度拆分：

| 难度 | 数量 |
|------|-----:|
| easy | 9 |
| medium | 35 |
| hard | 10 |
| extra | 17 |

这些题的选择错误模式：

| 模式 | 数量 | 解释 |
|------|-----:|------|
| correct_minority | 50 | 正确候选存在，但输给更大的错误结果簇 |
| merged_vote_would_fix | 16 | 简单合并投票已经能选对 |
| correct_tied_largest_but_lost_tiebreak | 5 | 正确结果簇并列最大，但 tie-break 输了 |

正确结果簇在当前 vote-style ranking 下的排名：

| 正确结果簇排名 | 数量 |
|---------------|-----:|
| 1 | 16 |
| 2 | 42 |
| 3 | 11 |
| 4 | 2 |

这是最核心的归因结果。只有 **5** 题是纯 tie-break 问题；大多数可追回样本需要识别“不是多数派但正确”的候选。

## 为什么简单合并投票不够

把 SFT 和 DPO candidates 直接合并后再投票，不能吃掉 oracle gap：

| 选择器变体 | Correct | EX |
|------------|--------:|---:|
| DPO SC-12 | 775 | 74.95% |
| merged vote, empty results unique | 773 | 74.76% |
| merged vote, empty results clustered | 770 | 74.47% |
| SFT + DPO oracle | 842 | 81.43% |

原因是：增加候选不只会增加正确 SQL，也会增加可执行的错误 SQL 和错误结果簇。错误候选也会抱团。

所以 verifier 不能只是：

```text
更多候选 + majority vote
```

而需要候选级 correctness signal。

## 候选级信号

在所有可执行的 unique candidates 中，正确候选和错误候选有一些粗粒度差异：

| 特征 | 正确候选平均值 | 错误候选平均值 |
|------|---------------:|---------------:|
| occurrence count | 8.11 | 3.94 |
| minimum sample rank | 2.36 | 3.44 |
| SQL length | 119.24 | 159.98 |
| SQL token count | 12.41 | 15.17 |
| result size | 11.66 | 32.98 |
| vote cluster occurrence size | 20.15 | 10.82 |
| number of JOINs | 0.65 | 1.01 |
| tables mentioned | 1.67 | 1.99 |

这些特征不能直接当规则用，但适合做 learned selector / reranker 的 baseline features。

直觉上，正确候选往往：

- 出现次数更多
- 采样 rank 更靠前
- SQL 更短
- JOIN 更少
- 结果簇更大

但 minority-correct 样本恰恰说明：这些信号只能辅助，不能简单等价于正确性。

## Pairwise 细化归因

上面的统计回答的是：

```text
哪些题有可追回空间？
正确候选是不是少数派？
```

但它还没有回答：

```text
错误 winner 具体错在哪里？
oracle correct candidate 和 wrong selected SQL 差在哪？
```

因此进一步构造 `wrong selected SQL` vs `oracle correct SQL` 的成对归因。当前只看 DPO 错但 oracle 对的 **71** 个 recoverable cases。

每个样本固定一对：

```text
wrong_sql   = 当前 DPO SC-12 选中的错误 SQL
correct_sql = 候选池中执行正确的代表 SQL
```

如果同一题有多个正确候选，优先选择：

```text
结果簇更大 → 采样 rank 更靠前 → SQL 更短
```

输出文件：

```text
outputs/spider/diagnostics/sc_n12/recoverable_pairs.jsonl
outputs/spider/diagnostics/sc_n12/pair_attribution_summary.json
```

### Clause diff 统计

在 71 个 recoverable pairs 中，wrong-vs-correct 的结构差异如下：

| Diff tag | Count |
|----------|------:|
| SELECT | 30 |
| JOIN_CONDITION | 24 |
| GROUP_BY | 23 |
| WHERE_CONDITION | 20 |
| TABLE_SELECTION | 19 |
| COLUMN_ORDER | 10 |
| LITERAL_VALUE | 10 |
| AGGREGATION | 7 |
| ORDER_BY | 7 |
| VALUE_ONLY_DIFF | 7 |
| DISTINCT | 4 |
| FILTER_VALUE | 4 |
| SET_OP | 4 |
| SUBQUERY | 4 |
| HAVING | 3 |
| LITERAL_CASE | 3 |
| SORT_DIRECTION | 2 |

这些标签不是互斥的。一条 pair 可能同时有 `SELECT`、`GROUP_BY`、`JOIN_CONDITION` 等多个差异。

### 语义错误类型

将 clause diff 合并成更接近 verifier 任务的语义类别：

| Semantic tag | Count |
|--------------|------:|
| PROJECTION | 40 |
| FILTER_OR_VALUE_GROUNDING | 29 |
| JOIN_KEY_OR_PATH | 24 |
| GROUPING | 23 |
| TABLE_SELECTION | 19 |
| AGGREGATION | 7 |
| SORT_OR_EXTREME_VALUE | 7 |
| NESTED_QUERY | 4 |
| SET_OPERATION | 4 |
| SORT_DIRECTION | 2 |

这说明 recoverable gap 不是单一错误类型造成的。最主要的是：

```text
投影列 / 列顺序
过滤条件和值 grounding
JOIN key / join path
GROUP BY
表选择
```

因此 verifier 的候选级特征不能只看执行结果簇大小，还需要理解 SQL 结构和 question-schema grounding。

### Source / rank 信号

在这 71 个 recoverable pairs 中：

| 信号 | 统计 |
|------|------|
| correct candidate 来自 SFT | 54 |
| correct candidate 来自 DPO | 28 |
| wrong selected 来自 DPO | 71 |
| wrong selected 也出现在 SFT 候选中 | 42 |

正确候选的最小采样 rank：

| rank | Count |
|------|------:|
| 0 | 12 |
| 1 | 23 |
| 2 | 13 |
| 3 | 9 |
| 4 | 2 |
| 5 | 2 |
| 6 | 5 |
| 8 | 4 |
| 9 | 1 |

错误 winner 的最小采样 rank：

| rank | Count |
|------|------:|
| 0 | 56 |
| 1 | 10 |
| 2 | 2 |
| 3 | 1 |
| 4 | 1 |
| 8 | 1 |

这个结果很关键：wrong selected 往往是 rank 0 或高频候选；correct candidate 经常不是最高 rank。也就是说，rank/频次是强信号，但会系统性压制 minority-correct。

### Pairwise 归因的意义

现在的逻辑链条更完整：

```text
Oracle gap
  ↓
DPO 错但 oracle 可修：71
  ↓
选择失败模式：多数是 correct_minority
  ↓
wrong-vs-correct pairwise diff
  ↓
知道 verifier 需要识别哪些 SQL 结构差异
```

这一步把问题从：

```text
模型选错了
```

细化为：

```text
模型经常选中高频但 projection / filter / join / group 错的 SQL；
正确 SQL 往往存在，但 rank 更靠后或票数更少。
```

这就是后续 learned verifier / reranker 的直接训练信号。

## 评估口径 caveat

有 4 个样本中，保存的 `results.json` 标记 SFT/DPO SC 为正确，但严格候选级 oracle 没有找到正确候选。这主要是评估口径差异，而不是模型本身的新现象。

典型来源包括：

- `wta_1`：按 player name / id 聚合的变体
- `student_transcripts_tracking`：不同 `ORDER BY` 列在当前数据上可能表现接近
- `orchestra`：`ORDER BY COUNT(*) ASC` vs `DESC` 的 evaluator / 数据边界情况

后续训练 learned verifier 前，必须明确 label contract：

1. **Execution-strict label**：候选执行结果严格匹配 gold 执行结果；当 gold 有 `ORDER BY` 时按顺序比较。
2. **Spider-evaluator label**：候选是否被当前 `results.json` 使用的评估器判为正确。

当前 diagnostics 使用的是 execution-strict label，并且当 gold SQL 包含 `ORDER BY` 时启用 order-sensitive 比较。

## 对 Verifier 下一步的含义

后续 verifier 应该针对三个具体问题：

1. **SFT-vs-DPO conflict selection**：处理 76 个 SFT/DPO 冲突样本。
2. **Minority-correct reranking**：处理 37 个两个 SC 都错但候选池有正确 SQL 的样本。
3. **Label-noise control**：区分真正语义正确、执行等价、以及数据偶然正确。

第一版 offline baseline 应该直接使用：

```text
outputs/spider/diagnostics/sc_n12/candidate_labels.jsonl
outputs/spider/diagnostics/sc_n12/item_diagnostics.jsonl
```

建议先用以下特征做轻量 selector：

- policy source：来自 SFT / DPO / 两者都有
- sample rank
- occurrence count
- result cluster size
- executable / empty result
- SQL 长度
- SQL 结构特征：JOIN、GROUP BY、ORDER BY、LIMIT、subquery、aggregation
- question-SQL token overlap

这一步的目标很明确：

> 在进入 verifier distillation 或 RL 之前，先确认一个 execution-aware selector 到底能追回多少 81.43% oracle gap。
