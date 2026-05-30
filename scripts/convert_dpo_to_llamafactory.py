#!/usr/bin/env python3
"""将 prepare_dpo.py 的输出转换为 LLaMA-Factory DPO 格式（JSONL）。

LLaMA-Factory 要求:
  - Alpaca preference 数据使用 instruction/input/chosen/rejected
  - system 字段可选；这里保留 prepare_dpo.py 的 system prompt
  - dataset_info.json 需加 "ranking": true，并映射 chosen/rejected

用法:
    python scripts/convert_dpo_to_llamafactory.py \
        --input data_processed/dpo_pairs.json \
        --output data/dpo_data.jsonl
"""

import argparse
import json
from pathlib import Path


def convert(input_path: str, output_path: str) -> None:
    with open(input_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} pairs from {input_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            system_content = item["messages"][0]["content"]
            user_content = item["messages"][1]["content"]
            record = {
                "instruction": user_content,
                "input": "",
                "chosen": item["chosen"],
                "rejected": item["rejected"],
                "system": system_content,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(data)} pairs to {output_path}")
    print(f"Format: JSONL {{instruction, input, chosen, rejected, system}}")
    print()
    print("在 LLaMA-Factory data/dataset_info.json 中注册:")
    print("  \"spider_dpo\": {")
    print(f"    \"file_name\": \"{output_path.name}\",")
    print("    \"formatting\": \"alpaca\",")
    print("    \"ranking\": true,")
    print("    \"columns\": {")
    print("      \"prompt\": \"instruction\",")
    print("      \"query\": \"input\",")
    print("      \"chosen\": \"chosen\",")
    print("      \"rejected\": \"rejected\",")
    print("      \"system\": \"system\"")
    print("    }")
    print("  }")


def main():
    parser = argparse.ArgumentParser(description="Convert DPO pairs to LLaMA-Factory format")
    parser.add_argument("--input", type=str, default="data_processed/dpo_pairs.json",
                        help="Input from prepare_dpo.py")
    parser.add_argument("--output", type=str, default="data/dpo_data.jsonl",
                        help="Output for LLaMA-Factory (.jsonl)")
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
