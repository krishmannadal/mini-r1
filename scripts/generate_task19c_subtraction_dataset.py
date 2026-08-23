"""
generate_task19c_subtraction_dataset.py - Deterministic Dataset Generator for Task 19C.

Generates:
1. Training set (12,000 items) sampled from 1,000 unique [50, 99] subtraction pairs.
2. Dev set (1,200 items) sampled from the same 1,000 unique [50, 99] pairs.
3. E1 IID Holdout (500 items) sampled strictly from 275 held-out [50, 99] pairs.
4. E2 Held-Out 2-Digit (500 items) from [2, 49] pairs.
5. E3 OOD 3-Digit (500 items) from [101, 199] pairs (1-2 digit answers).
6. E3-Ext OOD 3-Digit Answers (500 items) with 3-digit answers.
"""

import hashlib
import json
import os
import random
from collections import Counter
from typing import Any


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def format_compact_example(example_id: str, a: int, b: int) -> dict[str, Any]:
    res = a - b
    prompt = f"Question: Calculate: {a} - {b}\n"
    completion = f"<think>\n{a} - {b} = {res}\n</think>\n<answer>{res}</answer>"
    return {
        "example_id": example_id,
        "prompt": prompt,
        "completion": completion,
        "metadata": {
            "op": "-",
            "lhs": a,
            "rhs": b,
            "final_answer": str(res),
            "ans_digits": len(str(res)),
        },
    }


def format_eval_example(example_id: str, a: int, b: int) -> dict[str, Any]:
    res = a - b
    prompt = f"Question: Calculate: {a} - {b}\n"
    return {
        "example_id": example_id,
        "prompt": prompt,
        "expression": f"{a} - {b}",
        "lhs": a,
        "rhs": b,
        "final_answer": str(res),
        "ans_digits": len(str(res)),
    }


def main():
    seed = 42
    rng = random.Random(seed)
    out_dir = "data/subtraction_sft"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print(" TASK 19C: DETERMINISTIC SUBTRACTION DATASET GENERATOR")
    print(f" Seed: {seed} | Output Dir: {out_dir}")
    print("=" * 80)

    # 1. Enumerate all unique pairs in [50, 99] with A >= B
    all_50_99_pairs = []
    for a in range(50, 100):
        for b in range(50, a + 1):
            all_50_99_pairs.append((a, b))

    print(f"\nTotal unique pairs in [50, 99] with A >= B: {len(all_50_99_pairs)}")

    # Shuffle deterministically
    rng.shuffle(all_50_99_pairs)

    # Partition: 1,000 for Train/Dev pool, 275 strictly held out for E1
    train_pool_pairs = all_50_99_pairs[:1000]
    e1_holdout_pairs = all_50_99_pairs[1000:]

    print(f"  - Train/Dev Pair Pool: {len(train_pool_pairs)} unique pairs")
    print(f"  - E1 IID Holdout Pair Pool: {len(e1_holdout_pairs)} unique pairs")

    train_pool_set = set(train_pool_pairs)
    e1_holdout_set = set(e1_holdout_pairs)
    assert len(train_pool_set & e1_holdout_set) == 0, "Leakage between train pool and E1 holdout!"

    # 2. Generate Train Set (12,000 items)
    train_examples = []
    for i in range(12000):
        pair = train_pool_pairs[i % len(train_pool_pairs)]
        train_examples.append(pair)
    rng.shuffle(train_examples)

    train_data = [
        format_compact_example(f"sub_train_{i+1:05d}", a, b)
        for i, (a, b) in enumerate(train_examples)
    ]

    train_path = os.path.join(out_dir, "train.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")
    print(f"\n[Generated] {train_path} ({len(train_data)} items)")

    # 3. Generate Dev Set (1,200 items)
    dev_examples = []
    for i in range(1200):
        pair = rng.choice(train_pool_pairs)
        dev_examples.append(pair)

    dev_data = [
        format_compact_example(f"sub_dev_{i+1:04d}", a, b)
        for i, (a, b) in enumerate(dev_examples)
    ]

    dev_path = os.path.join(out_dir, "dev.jsonl")
    with open(dev_path, "w", encoding="utf-8") as f:
        for ex in dev_data:
            f.write(json.dumps(ex) + "\n")
    print(f"[Generated] {dev_path} ({len(dev_data)} items)")

    # 4. Generate E1: IID Holdout (500 items from e1_holdout_pairs)
    e1_examples = []
    # Use all 275 unique held-out pairs, plus sampling from them to reach 500
    e1_examples.extend(e1_holdout_pairs)
    while len(e1_examples) < 500:
        e1_examples.append(rng.choice(e1_holdout_pairs))
    rng.shuffle(e1_examples)

    e1_data = [
        format_eval_example(f"e1_iid_{i+1:04d}", a, b)
        for i, (a, b) in enumerate(e1_examples)
    ]

    e1_path = os.path.join(out_dir, "eval_e1_iid_holdout.json")
    with open(e1_path, "w", encoding="utf-8") as f:
        json.dump(e1_data, f, indent=2)
    print(f"[Generated] {e1_path} ({len(e1_data)} items, {len(set(e1_examples))} unique pairs)")

    # 5. Generate E2: Held-Out 2-Digit [2, 49], A >= B (500 items)
    all_2_49_pairs = []
    for a in range(2, 50):
        for b in range(2, a + 1):
            all_2_49_pairs.append((a, b))
    rng.shuffle(all_2_49_pairs)

    e2_pairs = all_2_49_pairs[:500]
    e2_data = [
        format_eval_example(f"e2_2digit_{i+1:04d}", a, b)
        for i, (a, b) in enumerate(e2_pairs)
    ]

    e2_path = os.path.join(out_dir, "eval_e2_heldout_2digit.json")
    with open(e2_path, "w", encoding="utf-8") as f:
        json.dump(e2_data, f, indent=2)
    print(f"[Generated] {e2_path} ({len(e2_data)} items, all unique)")

    # 6. Generate E3: OOD 3-Digit [101, 199], A >= B (500 items, answer in [0, 98])
    all_101_199_pairs = []
    for a in range(101, 200):
        for b in range(101, a + 1):
            all_101_199_pairs.append((a, b))
    rng.shuffle(all_101_199_pairs)

    e3_pairs = all_101_199_pairs[:500]
    e3_data = [
        format_eval_example(f"e3_3digit_{i+1:04d}", a, b)
        for i, (a, b) in enumerate(e3_pairs)
    ]

    e3_path = os.path.join(out_dir, "eval_e3_ood_3digit.json")
    with open(e3_path, "w", encoding="utf-8") as f:
        json.dump(e3_data, f, indent=2)
    print(f"[Generated] {e3_path} ({len(e3_data)} items, all unique, answers in [0, 98])")

    # 7. Generate E3-Ext: OOD 3-Digit with 3-Digit Answers (500 items)
    # A in [200, 300], B in [50, 99] -> A - B in [101, 250]
    all_3digit_ans_pairs = []
    for a in range(200, 301):
        for b in range(50, 100):
            if a - b >= 100:
                all_3digit_ans_pairs.append((a, b))
    rng.shuffle(all_3digit_ans_pairs)

    e3_ext_pairs = all_3digit_ans_pairs[:500]
    e3_ext_data = [
        format_eval_example(f"e3_ext_3d_ans_{i+1:04d}", a, b)
        for i, (a, b) in enumerate(e3_ext_pairs)
    ]

    e3_ext_path = os.path.join(out_dir, "eval_e3_ext_3digit_ans.json")
    with open(e3_ext_path, "w", encoding="utf-8") as f:
        json.dump(e3_ext_data, f, indent=2)
    print(f"[Generated] {e3_ext_path} ({len(e3_ext_data)} items, answers >= 100)")

    # 8. Complete Overlap Verification Audit
    print("\n--- Running Rigorous Overlap Audit ---")
    train_pairs_set = set((ex["metadata"]["lhs"], ex["metadata"]["rhs"]) for ex in train_data)
    dev_pairs_set = set((ex["metadata"]["lhs"], ex["metadata"]["rhs"]) for ex in dev_data)

    e1_set = set((ex["lhs"], ex["rhs"]) for ex in e1_data)
    e2_set = set((ex["lhs"], ex["rhs"]) for ex in e2_data)
    e3_set = set((ex["lhs"], ex["rhs"]) for ex in e3_data)
    e3_ext_set = set((ex["lhs"], ex["rhs"]) for ex in e3_ext_data)

    e1_overlap_train = len(e1_set & train_pairs_set)
    e1_overlap_dev = len(e1_set & dev_pairs_set)
    e2_overlap_train = len(e2_set & train_pairs_set)
    e2_overlap_dev = len(e2_set & dev_pairs_set)
    e3_overlap_train = len(e3_set & train_pairs_set)
    e3_overlap_dev = len(e3_set & dev_pairs_set)
    e3_ext_overlap_train = len(e3_ext_set & train_pairs_set)

    print(f"  E1 Overlap with Train: {e1_overlap_train} | with Dev: {e1_overlap_dev}")
    print(f"  E2 Overlap with Train: {e2_overlap_train} | with Dev: {e2_overlap_dev}")
    print(f"  E3 Overlap with Train: {e3_overlap_train} | with Dev: {e3_overlap_dev}")
    print(f"  E3-Ext Overlap with Train: {e3_ext_overlap_train}")

    assert e1_overlap_train == 0 and e1_overlap_dev == 0, "E1 has overlap!"
    assert e2_overlap_train == 0 and e2_overlap_dev == 0, "E2 has overlap!"
    assert e3_overlap_train == 0 and e3_overlap_dev == 0, "E3 has overlap!"
    assert e3_ext_overlap_train == 0, "E3-Ext has overlap!"
    print(">>> OVERLAP AUDIT PASSED: EXACT 0 OVERLAP CONFIRMED ACROSS ALL EVALUATION SETS! <<<")

    # 9. Digit length statistics
    def digit_counts(dataset: list[dict[str, Any]]) -> dict[int, int]:
        c = Counter(ex.get("ans_digits", ex.get("metadata", {}).get("ans_digits")) for ex in dataset)
        return dict(c)

    print("\n--- Answer Digit-Length Distributions ---")
    print(f"  Train:  {digit_counts(train_data)}")
    print(f"  E1:     {digit_counts(e1_data)}")
    print(f"  E2:     {digit_counts(e2_data)}")
    print(f"  E3:     {digit_counts(e3_data)}")
    print(f"  E3-Ext: {digit_counts(e3_ext_data)}")

    # 10. Write Manifest
    manifest = {
        "dataset_name": "Task 19C Clean Subtraction SFT",
        "seed": seed,
        "splits": {
            "train": {
                "file": train_path,
                "count": len(train_data),
                "unique_pairs": len(train_pairs_set),
                "sha256": compute_sha256(train_path),
                "answer_digit_distribution": digit_counts(train_data),
            },
            "dev": {
                "file": dev_path,
                "count": len(dev_data),
                "unique_pairs": len(dev_pairs_set),
                "sha256": compute_sha256(dev_path),
                "answer_digit_distribution": digit_counts(dev_data),
            },
            "e1_iid_holdout": {
                "file": e1_path,
                "count": len(e1_data),
                "unique_pairs": len(e1_set),
                "sha256": compute_sha256(e1_path),
                "answer_digit_distribution": digit_counts(e1_data),
            },
            "e2_heldout_2digit": {
                "file": e2_path,
                "count": len(e2_data),
                "unique_pairs": len(e2_set),
                "sha256": compute_sha256(e2_path),
                "answer_digit_distribution": digit_counts(e2_data),
            },
            "e3_ood_3digit": {
                "file": e3_path,
                "count": len(e3_data),
                "unique_pairs": len(e3_set),
                "sha256": compute_sha256(e3_path),
                "answer_digit_distribution": digit_counts(e3_data),
            },
            "e3_ext_3digit_ans": {
                "file": e3_ext_path,
                "count": len(e3_ext_data),
                "unique_pairs": len(e3_ext_set),
                "sha256": compute_sha256(e3_ext_path),
                "answer_digit_distribution": digit_counts(e3_ext_data),
            },
        },
        "overlap_audit": {
            "e1_overlap": 0,
            "e2_overlap": 0,
            "e3_overlap": 0,
            "e3_ext_overlap": 0,
            "status": "PASSED_ZERO_OVERLAP",
        },
    }

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[Saved] Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
