"""
Mini-R1 -- Dataset Inspection Script
====================================
Inspects real online datasets using Hugging Face datasets in streaming mode:
1. HuggingFaceTB/smollm-corpus (subsets: cosmopedia-v2, fineweb-edu-dedup, python-edu)
2. openai/gsm8k (main config: train and test splits)

Features:
- Streaming-first: inspects schema and samples without downloading the full corpus.
- Verifies GSM8K multi-step reasoning structure and splits.
- Reports fields, sample text, and split statistics.

Usage:
    python scripts/data_inspect.py
"""

from __future__ import annotations

import sys
from typing import Any
from datasets import load_dataset


def separator(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def inspect_smollm_corpus() -> dict[str, Any]:
    """
    Inspect HuggingFaceTB/smollm-corpus subsets in streaming mode.
    Streams individual shard headers to report schema without full multi-GB download.
    """
    separator("1. HuggingFaceTB/smollm-corpus (Pretraining Source)")
    subset_shards = {
        "cosmopedia-v2": "cosmopedia-v2/train-00000-of-00104.parquet",
        "fineweb-edu-dedup": "fineweb-edu-dedup/train-00000-of-00234.parquet",
        "python-edu": "python-edu/train-00000-of-00002.parquet",
    }
    subset_info = {}

    for subset, shard in subset_shards.items():
        print(f"\n  --- Subset: {subset} ---")
        try:
            ds_stream = load_dataset(
                "HuggingFaceTB/smollm-corpus",
                data_files={"train": shard},
                split="train",
                streaming=True,
            )
            sample = next(iter(ds_stream))
            fields = list(sample.keys())
            text_field = "text" if "text" in sample else fields[0]
            sample_text = sample[text_field]
            preview = sample_text[:250].replace("\n", " ")

            print(f"  Available fields : {fields}")
            print(f"  Primary field    : '{text_field}'")
            print(f"  Sample length    : {len(sample_text)} characters")
            print(f"  Sample preview   : {preview}...\n")

            subset_info[subset] = {
                "fields": fields,
                "text_field": text_field,
                "sample_preview": preview,
            }
        except Exception as e:
            print(f"  [WARN] Failed to stream subset '{subset}': {e}")

    print("  [Rationale for Pretraining Subset Selection]")
    print("  - 'cosmopedia-v2': High-quality synthetic textbooks, tutorials, and structured explanations.")
    print("  - 'fineweb-edu-dedup': Educational reasoning and scientific web texts.")
    print("  -> 'cosmopedia-v2' is selected as our primary pretraining data source for Mini-R1")
    print("     because its structured, expository style establishes strong foundational grammar")
    print("     and step-by-step explanatory patterns ideal for a small language model.")

    return subset_info


def inspect_gsm8k() -> dict[str, Any]:
    """
    Inspect openai/gsm8k (main config).
    Validates train and test split sizes, fields, and multi-step reasoning format.
    """
    separator("2. openai/gsm8k (Reasoning SFT & Evaluation Source)")
    print("  Loading openai/gsm8k (config='main')...")

    dataset = load_dataset("openai/gsm8k", "main")
    train_ds = dataset["train"]
    test_ds = dataset["test"]

    print(f"  Train split size : {len(train_ds)} examples")
    print(f"  Test split size  : {len(test_ds)} examples (strictly reserved for evaluation)")
    print(f"  Fields           : {train_ds.column_names}")

    sample = train_ds[0]
    question = sample["question"]
    answer = sample["answer"]

    print("\n  --- GSM8K Train Sample Record [0] ---")
    print(f"  Question:\n    {question}")
    print(f"\n  Full Worked Answer (Multi-step Reasoning):\n{answer}")

    has_calc_annotations = "<<" in answer and ">>" in answer
    has_final_answer_tag = "####" in answer

    print("\n  --- Reasoning Structure Verification ---")
    print(f"  Contains internal calculation steps (<<...>>): {has_calc_annotations}")
    print(f"  Contains final answer marker (####)           : {has_final_answer_tag}")
    assert has_final_answer_tag, "GSM8K answer does not contain final answer marker '####'"

    return {
        "train_size": len(train_ds),
        "test_size": len(test_ds),
        "fields": train_ds.column_names,
    }


def main() -> None:
    print("+====================================================================+")
    print("|            Mini-R1 -- Dataset Inspection & Verification            |")
    print("+====================================================================+")

    try:
        smollm_info = inspect_smollm_corpus()
        gsm8k_info = inspect_gsm8k()

        separator("Summary")
        print("  [OK] Online datasets successfully accessed and verified.")
        print(f"  Pretraining candidates : {list(smollm_info.keys())}")
        print(f"  Reasoning train set    : {gsm8k_info.get('train_size')} examples")
        print(f"  Reasoning test set     : {gsm8k_info.get('test_size')} examples (strictly isolated)")
    except Exception as e:
        print(f"\n[FAIL] Dataset inspection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
