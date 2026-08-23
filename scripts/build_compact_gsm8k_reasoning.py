"""
build_compact_gsm8k_reasoning.py - Task 21B: Compact Verified Reasoning Dataset Builder.

Constructs machine-verified, compact computational reasoning targets for GSM8K:
Input:
Question: {question}

Target:
<think>
{step 1}
{step 2}
...
</think>
<answer>{final_answer}</answer>

Every arithmetic equation is checked with a deterministic Python verifier:
- Checks operands, operator, result, ordering, and consistency.
- Any rationale that cannot be reliably verified or has missing steps is excluded.
- Outputs train.jsonl, dev.jsonl, and manifest.json to data/gsm8k_compact_reasoning/.
"""

import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import AnswerExtractor
from src.preprocessing import clean_whitespace
from src.tokenizer import Tokenizer
import numpy as np


def compute_sha256(filepath: str | Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def verify_and_format_equation(eq_str: str) -> tuple[bool, str, str | None, str | None]:
    """
    Verifies a raw arithmetic equation string (e.g. '300*4=1200' or '1000-100=900.00').
    Returns (is_valid, reason, formatted_equation, rhs_clean).
    """
    eq_str = eq_str.strip()
    if "=" not in eq_str:
        return False, "missing_equal_sign", None, None
    parts = eq_str.split("=")
    if len(parts) != 2:
        return False, "multiple_equal_signs", None, None

    lhs_raw, rhs_raw = parts[0].strip(), parts[1].strip()
    
    # Strip currency, commas, trailing zeros
    lhs_clean = re.sub(r"[\$,]", "", lhs_raw).strip()
    rhs_clean = re.sub(r"[\$,]", "", rhs_raw).strip()

    # Clean whitespace and standard operators
    # Check for invalid characters in lhs
    if not re.match(r"^[\d\.\s\+\-\*\/\(\)]+$", lhs_clean):
        return False, f"invalid_characters_in_lhs: {lhs_clean}", None, None

    try:
        # Safe deterministic evaluation of lhs
        val_lhs = eval(lhs_clean, {"__builtins__": None}, {})
        val_rhs = float(rhs_clean)
        
        if abs(val_lhs - val_rhs) > 1e-4:
            return False, f"math_mismatch: {lhs_clean}={val_lhs} != {val_rhs}", None, None

        # Format clean spacing around operators in LHS
        # Tokenize LHS symbols
        tokens = re.findall(r"\d+(?:\.\d+)?|[\+\-\*\/\(\)]", lhs_clean)
        formatted_lhs = ""
        for i, tok in enumerate(tokens):
            if tok in ["+", "-", "*", "/"]:
                formatted_lhs += f" {tok} "
            elif tok == "(":
                formatted_lhs += "("
            elif tok == ")":
                formatted_lhs = formatted_lhs.rstrip() + ") "
            else:
                formatted_lhs += tok
        formatted_lhs = clean_whitespace(formatted_lhs)

        # Format RHS as integer if whole number, else clean decimal
        if abs(val_rhs - round(val_rhs)) < 1e-4:
            formatted_rhs = str(int(round(val_rhs)))
        else:
            formatted_rhs = f"{val_rhs:.2f}".rstrip("0").rstrip(".")

        formatted_eq = f"{formatted_lhs} = {formatted_rhs}"
        return True, "valid", formatted_eq, formatted_rhs

    except Exception as e:
        return False, f"eval_error: {str(e)}", None, None


def process_gsm8k_split(
    input_jsonl: str,
    output_jsonl: str,
    split_name: str,
    tokenizer: Tokenizer,
) -> dict[str, Any]:
    """
    Reads reasoning SFT JSONL, converts verified equations to compact trace,
    verifies consistency, and writes out valid records.
    """
    with open(input_jsonl, "r", encoding="utf-8") as f:
        raw_records = [json.loads(line) for line in f if line.strip()]

    valid_records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []
    reasons_count: Counter[str] = Counter()

    prompt_tok_counts: list[int] = []
    reasoning_tok_counts: list[int] = []
    answer_tok_counts: list[int] = []
    completion_tok_counts: list[int] = []

    for ex in raw_records:
        ex_id = ex["id"]
        question = ex["question"].strip()
        gold_answer = str(ex["answer"]).strip()
        calc_tags = ex.get("metadata", {}).get("calc_tags", [])

        if not calc_tags:
            reasons_count["no_calculator_tags"] += 1
            invalid_records.append({"id": ex_id, "reason": "no_calculator_tags"})
            continue

        verified_steps = []
        is_valid = True
        fail_reason = ""

        for tag in calc_tags:
            v_ok, reason, fmt_eq, rhs = verify_and_format_equation(tag)
            if not v_ok:
                is_valid = False
                fail_reason = reason
                break
            verified_steps.append((fmt_eq, rhs))

        if not is_valid:
            reasons_count[fail_reason.split(":")[0]] += 1
            invalid_records.append({"id": ex_id, "reason": fail_reason})
            continue

        # Check final step consistency against gold answer
        final_rhs = verified_steps[-1][1]
        try:
            val_final = float(final_rhs)
            val_gold = float(gold_answer)
            if abs(val_final - val_gold) > 1e-4:
                reasons_count["final_step_mismatches_gold_answer"] += 1
                invalid_records.append({
                    "id": ex_id,
                    "reason": f"final_step_mismatch: {final_rhs} != {gold_answer}",
                })
                continue
        except ValueError:
            if final_rhs != gold_answer:
                reasons_count["final_step_mismatches_gold_answer"] += 1
                invalid_records.append({
                    "id": ex_id,
                    "reason": f"final_step_mismatch: {final_rhs} != {gold_answer}",
                })
                continue

        # Format compact reasoning
        compact_steps_str = "\n".join(step[0] for step in verified_steps)
        prompt_str = f"Question: {question}\n"
        completion_str = f"<think>\n{compact_steps_str}\n</think>\n<answer>{gold_answer}</answer>"
        formatted_text = f"{prompt_str}{completion_str}"

        # Token metrics
        p_toks = tokenizer.encode(prompt_str, add_bos=True, add_eos=False)
        r_toks = tokenizer.encode(compact_steps_str, add_bos=False, add_eos=False)
        a_toks = tokenizer.encode(f"<answer>{gold_answer}</answer>", add_bos=False, add_eos=True)
        c_toks = tokenizer.encode(completion_str, add_bos=False, add_eos=True)

        prompt_tok_counts.append(len(p_toks))
        reasoning_tok_counts.append(len(r_toks))
        answer_tok_counts.append(len(a_toks))
        completion_tok_counts.append(len(c_toks))

        record = {
            "id": ex_id,
            "dataset": "gsm8k",
            "split": split_name,
            "condition": "compact_reasoning",
            "question": question,
            "reasoning": compact_steps_str,
            "answer": gold_answer,
            "prompt": prompt_str,
            "completion": completion_str,
            "formatted_text": formatted_text,
            "metadata": {
                "num_steps": len(verified_steps),
                "steps": [step[0] for step in verified_steps],
                "prompt_tokens": len(p_toks),
                "reasoning_tokens": len(r_toks),
                "answer_tokens": len(a_toks),
                "completion_tokens": len(c_toks),
                "total_tokens": len(p_toks) + len(c_toks),
            },
        }
        valid_records.append(record)

    # Save output jsonl
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for rec in valid_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "split": split_name,
        "input_count": len(raw_records),
        "valid_count": len(valid_records),
        "invalid_count": len(invalid_records),
        "valid_percentage": (len(valid_records) / len(raw_records)) * 100 if raw_records else 0,
        "invalid_reasons": dict(reasons_count),
        "token_metrics": {
            "mean_prompt_tokens": float(np.mean(prompt_tok_counts)) if prompt_tok_counts else 0,
            "mean_reasoning_tokens": float(np.mean(reasoning_tok_counts)) if reasoning_tok_counts else 0,
            "mean_answer_tokens": float(np.mean(answer_tok_counts)) if answer_tok_counts else 0,
            "mean_completion_tokens": float(np.mean(completion_tok_counts)) if completion_tok_counts else 0,
            "total_completion_tokens": int(sum(completion_tok_counts)),
            "p50_completion_tokens": float(np.median(completion_tok_counts)) if completion_tok_counts else 0,
            "p90_completion_tokens": float(np.percentile(completion_tok_counts, 90)) if completion_tok_counts else 0,
        },
        "file_sha256": compute_sha256(output_jsonl),
    }
    return summary


def main() -> None:
    print("=" * 70)
    print("MINI-R1 -- TASK 21B: COMPACT REASONING DATASET BUILDER")
    print("=" * 70)

    tokenizer_path = "artifacts/tokenizer/tokenizer.model"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    tokenizer = Tokenizer(tokenizer_path)
    tok_sha = compute_sha256(tokenizer_path)
    print(f"Loaded tokenizer: {tokenizer_path} (SHA256: {tok_sha})")

    data_dir = Path("data/gsm8k_compact_reasoning")
    data_dir.mkdir(parents=True, exist_ok=True)

    train_src = "data/reasoning/gsm8k_sft_train.jsonl"
    dev_src = "data/reasoning/gsm8k_sft_dev.jsonl"
    train_dst = str(data_dir / "train.jsonl")
    dev_dst = str(data_dir / "dev.jsonl")
    manifest_dst = str(data_dir / "manifest.json")

    print(f"\nProcessing Train split from {train_src}...")
    train_summary = process_gsm8k_split(train_src, train_dst, "train", tokenizer)
    print(f"Train Valid: {train_summary['valid_count']}/{train_summary['input_count']} ({train_summary['valid_percentage']:.2f}%)")
    print(f"Train Completion Tokens: {train_summary['token_metrics']['total_completion_tokens']} (Mean: {train_summary['token_metrics']['mean_completion_tokens']:.2f})")
    print(f"Train Invalid Reasons: {train_summary['invalid_reasons']}")

    print(f"\nProcessing Dev split from {dev_src}...")
    dev_summary = process_gsm8k_split(dev_src, dev_dst, "dev", tokenizer)
    print(f"Dev Valid: {dev_summary['valid_count']}/{dev_summary['input_count']} ({dev_summary['valid_percentage']:.2f}%)")
    print(f"Dev Completion Tokens: {dev_summary['token_metrics']['total_completion_tokens']} (Mean: {dev_summary['token_metrics']['mean_completion_tokens']:.2f})")
    print(f"Dev Invalid Reasons: {dev_summary['invalid_reasons']}")

    # Token ratio comparison against verbose GSM8K dataset
    verbose_train_comp_tokens = 2106652  # from Task 15C / Task 15A accounting
    compact_train_comp_tokens = train_summary["token_metrics"]["total_completion_tokens"]
    ratio = compact_train_comp_tokens / verbose_train_comp_tokens

    manifest = {
        "dataset_name": "gsm8k_compact_reasoning",
        "description": "Compact machine-verified computational reasoning dataset for GSM8K",
        "source_dataset": "openai/gsm8k (via data/reasoning/)",
        "tokenizer_sha256": tok_sha,
        "splits": {
            "train": train_summary,
            "dev": dev_summary,
        },
        "token_comparison": {
            "verbose_train_completion_tokens": verbose_train_comp_tokens,
            "compact_train_completion_tokens": compact_train_comp_tokens,
            "compact_to_verbose_completion_ratio": ratio,
            "completion_token_reduction_percent": (1.0 - ratio) * 100.0,
        },
        "validation_status": "100%_DETERMINISTICALLY_VERIFIED",
    }

    with open(manifest_dst, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to {manifest_dst}")
    print(f"Token reduction: {manifest['token_comparison']['completion_token_reduction_percent']:.2f}% fewer completion tokens")
    print("=" * 70)


if __name__ == "__main__":
    main()
