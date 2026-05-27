# vLLM 推理加速

> 使用 vLLM 加速模型推理，解决批量评测速度慢的问题。

## 为什么需要加速

评测 1034 条 Spider dev 数据时，传统逐条推理方式存在严重瓶颈：

| 方式 | 问题 |
|------|------|
| 逐条调用 | 1034 次请求串行等待，GPU 利用率低 |
| 总耗时 | 可能需要数小时 |

## vLLM 核心原理

### PagedAttention

传统方法一次性加载所有 KV cache 到显存，显存利用率低且无法动态管理。

vLLM 将 KV cache 分块管理，像操作系统的分页内存一样：

```
传统：所有 KV cache 一次性占用整块显存
vLLM：分块管理，按需分配，不浪费
```

### Continuous Batching

传统 batch 是静态的，等所有请求都到达才开始推理：

```
请求1: |----推理----|
请求2:            |----推理----|
请求3:                        |----推理----|
GPU利用率: ████░░░░░░░░░░░░░░░░░░░░░

vLLM：动态添加新请求到正在运行的 batch
请求1: |----推理----|
请求2:   |----推理----|
请求3:     |----推理----|
GPU利用率: ████████████░░░░░░░░░░
```

### 并行生成

一次 API 调用处理多条序列，充分利用 GPU 并行能力。

## 项目实现

### 模型加载

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model=model_path,
    trust_remote_code=True,
    tensor_parallel_size=1,        # 单卡 RTX 4090
    gpu_memory_utilization=0.9,    # 占用 90% 显存
    max_model_len=8192,           # 最大上下文长度
)
tokenizer = llm.get_tokenizer()
```

### 批量推理

```python
# 一次生成多条 SQL
outputs = llm.generate(
    prompts,  # List[str]，多条 prompt
    SamplingParams(
        temperature=0,     # 确定性输出
        max_tokens=512,   # 最大生成长度
    ),
)
sqls = [extract_sql(o.outputs[0].text) for o in outputs]
```

### 完整流程

```python
# 1. 加载模型（一次性）
llm, tokenizer = load_model(model_path)

# 2. 构建批量 prompt
prompts = [build_prompt(schema, question) for schema, question in test_data]

# 3. 批量推理
sqls = batch_generate_sql(llm, tokenizer, prompts, max_new_tokens=512, temperature=0)

# 4. 并行评测
for sql, gold_sql in zip(sqls, gold_sqls):
    exec_score = execute_sql(db, sql) == execute_sql(db, gold_sql)
    exact_score = eval_exact_match(sql, gold_sql)
```

## 速度对比

| 场景 | 传统方式 | vLLM |
|------|----------|------|
| 1034 条数据 | ~1-2 小时 | ~5-10 分钟 |
| GPU 利用率 | ~30-50% | ~90%+ |
| 吞吐量 | ~10-20 req/s | ~100-200 req/s |

## Prompt 模板

针对不同阶段使用不同 Prompt：

| 阶段 | Prompt 特点 |
|------|-------------|
| **zeroshot** | 纯基座模型能力，需要明确规则约束输出 |
| **sft** | 训练过的模型，Prompt 更简洁 |
| **dpo** | 偏好对齐后，Prompt 进一步优化 |

```python
SYSTEM_PROMPTS = {
    "zeroshot": """You are an expert SQLite SQL generator.
CRITICAL RULES:
1. Output ONLY the raw SQL query
2. Do NOT use markdown code blocks
3. Do NOT include any explanation
4. Start directly with SELECT, WITH, INSERT, UPDATE, or DELETE
5. End with a semicolon""",

    "sft": """You are an expert SQLite SQL generator.
Given the database schema and question, output the correct SQL query.
CRITICAL RULES:
1. Output ONLY the raw SQL query - no markdown, no explanation
2. Start with SELECT/WITH/INSERT/UPDATE/DELETE, end with semicolon
3. Always use table_name.column_name when column name may be ambiguous""",

    "dpo": """Given the database schema and question, generate the correct SQL query.
Reminder:
- Output only raw SQL, no markdown or explanation
- Use table_name.column_name for disambiguation when needed""",
}
```

## SQL 提取

模型输出可能包含 markdown、解释等额外内容，需要提取纯 SQL：

```python
def extract_sql(text: str) -> str:
    # 1. 找第一条 SQL 语句
    for pattern in [r'SELECT', r'WITH', r'INSERT', r'UPDATE', r'DELETE']:
        if pattern in text.upper():
            # 提取直到分号
            match = re.search(rf'{pattern}[\s\S]+?;', text, re.IGNORECASE)
            if match:
                return match.group(0).strip()
    return ""
```

## 环境配置

```python
import os
os.environ["VLLM_USE_V1"] = "0"      # 使用 v0 引擎（兼容性）
os.environ["OMP_NUM_THREADS"] = "4"  # 限制线程数，避免资源竞争
```

## 依赖安装

```bash
pip install vllm
```

## 相关文件

- `scripts/evaluate_vllm.py` - 评测脚本
- `vendor/evaluation.py` - Spider 官方评测逻辑
- `vendor/process_sql.py` - SQL 解析工具