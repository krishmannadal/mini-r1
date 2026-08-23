"""
Mini-R1 -- Dataset Inspector Script
===================================
Inspects and reports dataset statistics, schemas, split counts,
and formatted sample records across GSM8K, SVAMP, and AQuA.

Usage:
    python scripts/inspect_datasets.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import ReasoningDatasetManager, DATASET_SOURCES


def separator(title: str) -> None:
    print(f"\n{'=' * 75}")
    print(f"  {title}")
    print(f"{'=' * 75}")


def main() -> None:
    print("+===========================================================================+")
    print("|          Mini-R1 -- Mathematical Reasoning Datasets Inspection            |")
    print("+===========================================================================+")

    manager = ReasoningDatasetManager(seed=42, val_ratio=0.1)

    # -------------------------------------------------------------------------
    # 1. Dataset Statistics & Split Breakdown
    # -------------------------------------------------------------------------
    separator("1. Dataset Statistics & Split Breakdown")
    stats = manager.get_statistics()

    print(f"  {'Dataset':<12} | {'Train':<10} | {'Validation':<12} | {'Test (Isolated)':<16} | {'Total':<10}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*16}-+-{'-'*10}")
    total_train, total_val, total_test, total_all = 0, 0, 0, 0
    for name, s in stats.items():
        print(f"  {name.upper():<12} | {s['train']:<10,d} | {s['validation']:<12,d} | {s['test']:<16,d} | {s['total']:<10,d}")
        total_train += s["train"]
        total_val += s["validation"]
        total_test += s["test"]
        total_all += s["total"]

    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*16}-+-{'-'*10}")
    print(f"  {'TOTAL':<12} | {total_train:<10,d} | {total_val:<12,d} | {total_test:<16,d} | {total_all:<10,d}")

    # -------------------------------------------------------------------------
    # 2. Schema and Raw Field Inspection
    # -------------------------------------------------------------------------
    separator("2. Schema & Raw Field Inspection")
    for name in DATASET_SOURCES:
        raw_ds = manager.load_raw(name)
        splits = list(raw_ds.keys())
        first_split = splits[0]
        fields = raw_ds[first_split].column_names

        print(f"\n  --- {name.upper()} ---")
        print(f"  Source Repo      : {DATASET_SOURCES[name]['path']}")
        print(f"  Available Splits : {splits}")
        print(f"  Available Fields : {fields}")

    # -------------------------------------------------------------------------
    # 3. Formatted Examples (Unified <think> / <answer> Schema)
    # -------------------------------------------------------------------------
    separator("3. Unified Reasoning Format Examples (<think> / <answer>)")
    for name in ["gsm8k", "svamp", "aqua"]:
        samples = manager.get_formatted_split(name, split="train", limit=1)
        if samples:
            s = samples[0]
            print(f"\n  [Dataset: {name.upper()}] (ID: {s['id']})")
            print(f"  --- Formatted Record ---")
            print(s["formatted_text"])
            print(f"  Extracted Answer: '{s['answer']}'\n")

    separator("Summary")
    print("  [OK] All 3 online datasets inspected and verified.")
    print("  Unified schema: Question -> <think> reasoning </think> -> <answer> final_answer </answer>")


if __name__ == "__main__":
    main()
