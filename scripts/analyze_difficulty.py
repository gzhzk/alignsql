#!/usr/bin/env python3
"""分析 Spider 数据集难度分布 —— 调用 alignsql 包。

Usage:
    python scripts/analyze_difficulty.py --dataset_dir dataset
"""

import argparse

from alignsql.data.spider import SpiderLoader


def main():
    parser = argparse.ArgumentParser(description="分析 Spider 数据集难度分布")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()

    loader = SpiderLoader(args.dataset_dir)
    data = loader.load_sft_data(split="train", max_samples=args.max_samples)

    distribution = loader.analyze_difficulty_distribution(data)
    print(f"\n难度分布 (总计 {distribution['total']} 条):")
    for diff in ["easy", "medium", "hard", "extra"]:
        info = distribution[diff]
        bar = "#" * int(info["pct"] / 2)
        print(f"  {diff:8s}: {info['count']:4d} ({info['pct']:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
