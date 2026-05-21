import argparse
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = "You are an expert SQL generator. Given a database schema, generate a correct SQLite SQL query to answer the user's question."


def build_prompt(schema_str: str, question: str) -> str:
    return f"""Database schema:
{schema_str}

Question: {question}
Only output the SQL query, no explanation."""


def load_schema(spider_dir: Path, db_id: str) -> str:
    tables_file = spider_dir / "tables.json"
    with open(tables_file, "r", encoding="utf-8") as f:
        tables_data = json.load(f)

    db_schema = None
    for db in tables_data:
        if db["db_id"] == db_id:
            db_schema = db
            break

    if db_schema is None:
        raise ValueError(f"Database {db_id} not found in tables.json")

    schema_parts = []
    for table in db_schema["table_names_original"]:
        columns = []
        for i, col in enumerate(db_schema["column_names_original"]):
            if col[0] == table:
                col_name = col[1]
                col_type = db_schema["column_types"][i] if i < len(db_schema["column_types"]) else "TEXT"
                columns.append(f"  - {col_name}: {col_type}")
        schema_parts.append(f"Table: {table}\n" + "\n".join(columns))

    return "\n\n".join(schema_parts)


def extract_sql(text: str) -> str:
    match = re.search(r"(SELECT|INSERT|UPDATE|DELETE|WITH).*?;(?:$|\n)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).strip().rstrip(";") + ";"
    text = text.strip().rstrip(";") + ";"
    if text.upper().startswith(("SELECT", "INSERT", "UPDATE", "DELETE", "WITH")):
        return text
    for line in text.split("\n"):
        line = line.strip()
        if line.upper().startswith(("SELECT", "INSERT", "UPDATE", "DELETE", "WITH")):
            return line.rstrip(";") + ";"
    return ""


def execute_sql(db_path: str, sql: str) -> tuple[bool, Any]:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.close()
        return True, result
    except Exception as e:
        return False, str(e)


def compare_results(result1: list, result2: list) -> bool:
    if len(result1) != len(result2):
        return False
    for r1, r2 in zip(result1, result2):
        if isinstance(r1, (int, float)) and isinstance(r2, (int, float)):
            if abs(float(r1) - float(r2)) > 0.0001:
                return False
        elif r1 != r2:
            return False
    return True


@torch.no_grad()
def generate_sql(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return extract_sql(response)


def load_model(model_path: str, device: str = "auto"):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


def evaluate(
    model_path: str,
    spider_dir: str,
    split: str = "dev",
    stage: str = "zeroshot",
    max_samples: int = -1,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
):
    spider_path = Path(spider_dir)
    data_file = spider_path / f"{split}.json"
    db_dir = spider_path / "database"

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if max_samples > 0:
        data = data[:max_samples]

    model, tokenizer = load_model(model_path)

    correct = 0
    total = len(data)
    results = []

    print(f"Evaluating {total} samples (stage: {stage})...")

    for i, item in enumerate(data):
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["query"]
        db_path = db_dir / db_id / f"{db_id}.sqlite"

        if not db_path.exists():
            db_path = db_dir / f"{db_id}.sqlite"

        try:
            schema_str = load_schema(spider_path, db_id)
            prompt = build_prompt(schema_str, question)
            pred_sql = generate_sql(model, tokenizer, prompt, max_new_tokens, temperature)

            if pred_sql:
                _, pred_result = execute_sql(str(db_path), pred_sql)
                _, gold_result = execute_sql(str(db_path), gold_sql)

                if isinstance(pred_result, list) and isinstance(gold_result, list):
                    is_correct = compare_results(pred_result, gold_result)
                else:
                    is_correct = False
            else:
                is_correct = False
                pred_result = "Failed to extract SQL"

            correct += 1 if is_correct else 0

            results.append({
                "index": i,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "pred_sql": pred_sql if pred_sql else "FAILED",
                "correct": is_correct,
            })

            if (i + 1) % 10 == 0 or i == total - 1:
                print(f"[{i + 1}/{total}] Current Accuracy: {correct}/{i + 1} = {100 * correct / (i + 1):.2f}%")

        except Exception as e:
            results.append({
                "index": i,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "pred_sql": f"ERROR: {str(e)}",
                "correct": False,
            })
            print(f"[{i + 1}/{total}] Error: {str(e)}")

    accuracy = correct / total * 100
    print(f"\n{'=' * 50}")
    print(f"Stage: {stage}")
    print(f"Final Accuracy: {correct}/{total} = {accuracy:.2f}%")
    print(f"{'=' * 50}")

    return results, accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluation on Spider dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--spider_dir", type=str, required=True, help="Path to Spider dataset")
    parser.add_argument("--stage", type=str, default="zeroshot",
                        choices=["zeroshot", "sft", "dpo"],
                        help="Experiment stage (zeroshot/sft/dpo)")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev"], help="Dataset split")
    parser.add_argument("--max_samples", type=int, default=-1, help="Max samples to evaluate (-1 for all)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--output_dir", type=str, default="experiments", help="Output directory for results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.stage
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results.json"

    results, accuracy = evaluate(
        model_path=args.model_path,
        spider_dir=args.spider_dir,
        split=args.split,
        stage=args.stage,
        max_samples=args.max_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy, "stage": args.stage, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()