#!/usr/bin/env python3
"""Spider 数据集 SFT 数据预处理 —— 调用 alignsql 包。

Usage:
    python scripts/prepare_sft.py --dataset_dir dataset --output data_processed/sft_data.json
"""

import argparse
from pathlib import Path

from alignsql.data.spider import SpiderLoader
from alignsql.utils.io import save_json


def main():
    parser = argparse.ArgumentParser(description="Spider SFT 数据预处理")
    parser.add_argument("--dataset_dir", type=str, default="dataset",
                        help="数据集目录")
    parser.add_argument("--output", type=str, default="data_processed/sft_data.json",
                        help="输出文件路径")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="最大样本数（-1 表示全部）")
    args = parser.parse_args()

    loader = SpiderLoader(args.dataset_dir)
    data = loader.load_sft_data(split="train", max_samples=args.max_samples)

    save_json(data, args.output)

    distribution = loader.analyze_difficulty_distribution(data)
    print(f"\nProcessed: {len(data)} samples")
    for diff, info in distribution.items():
        if diff != "total":
            print(f"  {diff}: {info['count']} ({info['pct']}%)")


if __name__ == "__main__":
    main()
