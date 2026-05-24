"""
Spider 数据集难度分类工具
基于 SQL 结构特征划分为 4 个难度级别
"""
import pyarrow.parquet as pq
import json
from pathlib import Path


def classify_difficulty(query: str) -> str:
    """
    基于 SQL 关键词的难度分类

    分类规则（Spider 官方标准）:
    - Easy: 简单 SELECT + WHERE，无聚合、无 GROUP BY
    - Medium: GROUP BY / ORDER BY / HAVING / 聚合函数
    - Hard: 多表 JOIN / 子查询
    - Extra: UNION / EXCEPT / INTERSECT / 深度嵌套子查询
    """
    query_upper = query.upper()

    # Extra: UNION / EXCEPT / INTERSECT
    if any(op in query_upper for op in ['UNION', 'EXCEPT', 'INTERSECT']):
        return 'extra'

    # Hard: 多表 JOIN 或子查询
    from_count = query_upper.count(' FROM ')
    if from_count > 1:
        return 'hard'

    # 检查子查询
    if query_upper.count('SELECT') > 1:
        return 'hard'

    # Medium: GROUP BY / ORDER BY / HAVING / 聚合函数
    if any(kw in query_upper for kw in ['GROUP BY', 'ORDER BY', 'HAVING']):
        return 'medium'

    if any(f in query_upper for f in ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(']):
        return 'medium'

    # Easy: 简单 SELECT
    return 'easy'


def analyze_parquet(parquet_path: str, output_path: str = None):
    """分析 parquet 文件的难度分布"""
    table = pq.read_table(parquet_path)

    counts = {'easy': 0, 'medium': 0, 'hard': 0, 'extra': 0}
    samples = {'easy': [], 'medium': [], 'hard': [], 'extra': []}
    results = []

    for i in range(table.num_rows):
        query = table.column('query')[i].as_py()
        question = table.column('question')[i].as_py()
        db_id = table.column('db_id')[i].as_py()

        difficulty = classify_difficulty(query)
        counts[difficulty] += 1

        if len(samples[difficulty]) < 3:
            samples[difficulty].append({
                'db_id': db_id,
                'question': question,
                'query': query
            })

        results.append({
            'db_id': db_id,
            'question': question,
            'query': query,
            'difficulty': difficulty
        })

    # 打印统计
    total = table.num_rows
    print(f"总样本数: {total}")
    print(f"\n难度分布:")
    print(f"  Easy:   {counts['easy']:4d} ({100*counts['easy']/total:.1f}%)")
    print(f"  Medium: {counts['medium']:4d} ({100*counts['medium']/total:.1f}%)")
    print(f"  Hard:   {counts['hard']:4d} ({100*counts['hard']/total:.1f}%)")
    print(f"  Extra:  {counts['extra']:4d} ({100*counts['extra']/total:.1f}%)")

    # 打印示例
    print(f"\n各难度示例:")
    for diff in ['easy', 'medium', 'hard', 'extra']:
        print(f"\n=== {diff.upper()} ===")
        for s in samples[diff]:
            print(f"Q: {s['question']}")
            print(f"SQL: {s['query']}\n")

    # 保存结果
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_path}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Spider 数据集难度分类')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入 parquet 文件路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出 JSON 文件路径（可选）')
    args = parser.parse_args()

    analyze_parquet(args.input, args.output)