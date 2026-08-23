"""
generate_task19d_mixed_subtraction.py - Task 19D: Mixed 2+3-Digit Subtraction Dataset Generator.

Generates:
- 12,000 train (6,000 from [50,99], 6,000 from [101,199])
- 1,200 dev (600 from each range)
- E1a: IID 2-digit holdout [50,99] (500 items)
- E1b: IID 3-digit holdout [101,199] (500 items)
- E2: OOD 2-digit [2,49] (500 items)
- E3: OOD 3-digit [200,299] (500 items, answer-matched [0,98])
- E4: Acid test 4-digit [1001,1999] (500 items)
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


def format_train_example(example_id: str, a: int, b: int) -> dict[str, Any]:
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
            "operand_digits": len(str(a)),
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
        "operand_digits": max(len(str(a)), len(str(b))),
    }


def enumerate_pairs(lo: int, hi: int) -> list[tuple[int, int]]:
    """All (a, b) with lo <= b <= a <= hi."""
    pairs = []
    for a in range(lo, hi + 1):
        for b in range(lo, a + 1):
            pairs.append((a, b))
    return pairs


def main():
    seed = 42
    rng = random.Random(seed)
    out_dir = "data/mixed_subtraction_sft"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print(" TASK 19D: MIXED 2+3-DIGIT SUBTRACTION DATASET GENERATOR")
    print(f" Seed: {seed} | Output Dir: {out_dir}")
    print("=" * 80)

    # =========================================================================
    # 1. Enumerate and partition [50, 99] pairs
    # =========================================================================
    all_2d_pairs = enumerate_pairs(50, 99)
    rng.shuffle(all_2d_pairs)
    print(f"\n[50, 99] total unique pairs: {len(all_2d_pairs)}")

    train_2d_pool = all_2d_pairs[:1000]
    holdout_2d_pool = all_2d_pairs[1000:]
    train_2d_set = set(train_2d_pool)
    holdout_2d_set = set(holdout_2d_pool)
    assert len(train_2d_set & holdout_2d_set) == 0

    print(f"  Train pool: {len(train_2d_pool)} | Holdout pool: {len(holdout_2d_pool)}")

    # =========================================================================
    # 2. Enumerate and partition [101, 199] pairs
    # =========================================================================
    all_3d_pairs = enumerate_pairs(101, 199)
    rng.shuffle(all_3d_pairs)
    print(f"\n[101, 199] total unique pairs: {len(all_3d_pairs)}")

    train_3d_pool = all_3d_pairs[:4000]
    holdout_3d_pool = all_3d_pairs[4000:]
    train_3d_set = set(train_3d_pool)
    holdout_3d_set = set(holdout_3d_pool)
    assert len(train_3d_set & holdout_3d_set) == 0

    print(f"  Train pool: {len(train_3d_pool)} | Holdout pool: {len(holdout_3d_pool)}")

    # =========================================================================
    # 3. Generate Training Set (12,000 = 6,000 + 6,000)
    # =========================================================================
    train_items_2d = []
    for i in range(6000):
        pair = train_2d_pool[i % len(train_2d_pool)]
        train_items_2d.append(pair)

    train_items_3d = []
    for i in range(6000):
        pair = train_3d_pool[i % len(train_3d_pool)]
        train_items_3d.append(pair)

    all_train_items = train_items_2d + train_items_3d
    rng.shuffle(all_train_items)

    train_data = [
        format_train_example(f"mix_train_{i+1:05d}", a, b)
        for i, (a, b) in enumerate(all_train_items)
    ]

    train_path = os.path.join(out_dir, "train.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")
    print(f"\n[Generated] {train_path} ({len(train_data)} items)")

    # =========================================================================
    # 4. Generate Dev Set (1,200 = 600 + 600)
    # =========================================================================
    dev_items = []
    for _ in range(600):
        dev_items.append(rng.choice(train_2d_pool))
    for _ in range(600):
        dev_items.append(rng.choice(train_3d_pool))
    rng.shuffle(dev_items)

    dev_data = [
        format_train_example(f"mix_dev_{i+1:04d}", a, b)
        for i, (a, b) in enumerate(dev_items)
    ]

    dev_path = os.path.join(out_dir, "dev.jsonl")
    with open(dev_path, "w", encoding="utf-8") as f:
        for ex in dev_data:
            f.write(json.dumps(ex) + "\n")
    print(f"[Generated] {dev_path} ({len(dev_data)} items)")

    # =========================================================================
    # 5. Generate E1a: IID 2-Digit Holdout [50, 99] (500 items)
    # =========================================================================
    e1a_items = list(holdout_2d_pool)
    while len(e1a_items) < 500:
        e1a_items.append(rng.choice(holdout_2d_pool))
    rng.shuffle(e1a_items)
    e1a_items = e1a_items[:500]

    e1a_data = [format_eval_example(f"e1a_iid2d_{i+1:04d}", a, b) for i, (a, b) in enumerate(e1a_items)]

    e1a_path = os.path.join(out_dir, "eval_e1a_iid_2digit.json")
    with open(e1a_path, "w", encoding="utf-8") as f:
        json.dump(e1a_data, f, indent=2)
    print(f"[Generated] {e1a_path} ({len(e1a_data)} items, {len(set(e1a_items))} unique)")

    # =========================================================================
    # 6. Generate E1b: IID 3-Digit Holdout [101, 199] (500 items)
    # =========================================================================
    e1b_items = list(holdout_3d_pool[:500])
    if len(e1b_items) < 500:
        while len(e1b_items) < 500:
            e1b_items.append(rng.choice(holdout_3d_pool))
    rng.shuffle(e1b_items)
    e1b_items = e1b_items[:500]

    e1b_data = [format_eval_example(f"e1b_iid3d_{i+1:04d}", a, b) for i, (a, b) in enumerate(e1b_items)]

    e1b_path = os.path.join(out_dir, "eval_e1b_iid_3digit.json")
    with open(e1b_path, "w", encoding="utf-8") as f:
        json.dump(e1b_data, f, indent=2)
    print(f"[Generated] {e1b_path} ({len(e1b_data)} items, {len(set(e1b_items))} unique)")

    # =========================================================================
    # 7. Generate E2: OOD 2-Digit [2, 49] (500 items)
    # =========================================================================
    all_e2_pairs = enumerate_pairs(2, 49)
    rng.shuffle(all_e2_pairs)
    e2_items = all_e2_pairs[:500]

    e2_data = [format_eval_example(f"e2_ood2d_{i+1:04d}", a, b) for i, (a, b) in enumerate(e2_items)]

    e2_path = os.path.join(out_dir, "eval_e2_ood_2digit.json")
    with open(e2_path, "w", encoding="utf-8") as f:
        json.dump(e2_data, f, indent=2)
    print(f"[Generated] {e2_path} ({len(e2_data)} items, all unique)")

    # =========================================================================
    # 8. Generate E3: OOD 3-Digit [200, 299] (500 items, answers [0, 98])
    # =========================================================================
    all_e3_pairs = enumerate_pairs(200, 299)
    rng.shuffle(all_e3_pairs)
    e3_items = all_e3_pairs[:500]

    e3_data = [format_eval_example(f"e3_ood3d_{i+1:04d}", a, b) for i, (a, b) in enumerate(e3_items)]

    e3_path = os.path.join(out_dir, "eval_e3_ood_3digit.json")
    with open(e3_path, "w", encoding="utf-8") as f:
        json.dump(e3_data, f, indent=2)
    print(f"[Generated] {e3_path} ({len(e3_data)} items, all unique)")

    # =========================================================================
    # 9. Generate E4: Acid Test 4-Digit [1001, 1999] (500 items)
    # =========================================================================
    e4_items = []
    e4_seen = set()
    while len(e4_items) < 500:
        a = rng.randint(1001, 1999)
        b = rng.randint(1001, 1999)
        if a < b:
            a, b = b, a
        key = (a, b)
        if key in e4_seen:
            continue
        e4_seen.add(key)
        e4_items.append((a, b))

    e4_data = [format_eval_example(f"e4_acid4d_{i+1:04d}", a, b) for i, (a, b) in enumerate(e4_items)]

    e4_path = os.path.join(out_dir, "eval_e4_acid_4digit.json")
    with open(e4_path, "w", encoding="utf-8") as f:
        json.dump(e4_data, f, indent=2)
    print(f"[Generated] {e4_path} ({len(e4_data)} items, all unique)")

    # =========================================================================
    # 10. Overlap Verification Audit
    # =========================================================================
    print("\n--- Running Rigorous Overlap Audit ---")
    train_pairs_set = set((ex["metadata"]["lhs"], ex["metadata"]["rhs"]) for ex in train_data)
    dev_pairs_set = set((ex["metadata"]["lhs"], ex["metadata"]["rhs"]) for ex in dev_data)

    eval_sets = {
        "E1a": set((ex["lhs"], ex["rhs"]) for ex in e1a_data),
        "E1b": set((ex["lhs"], ex["rhs"]) for ex in e1b_data),
        "E2": set((ex["lhs"], ex["rhs"]) for ex in e2_data),
        "E3": set((ex["lhs"], ex["rhs"]) for ex in e3_data),
        "E4": set((ex["lhs"], ex["rhs"]) for ex in e4_data),
    }

    all_pass = True
    for name, eset in eval_sets.items():
        ov_train = len(eset & train_pairs_set)
        ov_dev = len(eset & dev_pairs_set)
        status = "PASS" if ov_train == 0 and ov_dev == 0 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {name}: train_overlap={ov_train}, dev_overlap={ov_dev} [{status}]")

    assert all_pass, "OVERLAP AUDIT FAILED!"
    print(">>> OVERLAP AUDIT PASSED: EXACT 0 OVERLAP CONFIRMED! <<<")

    # =========================================================================
    # 11. Answer Digit-Length Distributions
    # =========================================================================
    def digit_counts(dataset: list[dict[str, Any]]) -> dict[int, int]:
        key = "ans_digits" if "ans_digits" in dataset[0] else None
        if key is None:
            key_meta = "ans_digits"
            c = Counter(ex.get("metadata", {}).get(key_meta) for ex in dataset)
        else:
            c = Counter(ex[key] for ex in dataset)
        return dict(sorted(c.items()))

    print("\n--- Answer Digit-Length Distributions ---")
    print(f"  Train: {digit_counts(train_data)}")
    for name, path in [("E1a", e1a_path), ("E1b", e1b_path), ("E2", e2_path), ("E3", e3_path), ("E4", e4_path)]:
        with open(path) as f:
            d = json.load(f)
        c = Counter(ex["ans_digits"] for ex in d)
        print(f"  {name}: {dict(sorted(c.items()))}")

    # =========================================================================
    # 12. Save Manifest
    # =========================================================================
    manifest = {
        "dataset_name": "Task 19D Mixed 2+3-Digit Subtraction SFT",
        "seed": seed,
        "training_composition": {
            "2_digit_component": {"range": "[50, 99]", "count": 6000, "unique_pairs": len(train_2d_set)},
            "3_digit_component": {"range": "[101, 199]", "count": 6000, "unique_pairs": min(len(train_3d_set), 4000)},
            "total": 12000,
        },
        "splits": {
            "train": {"file": train_path, "count": len(train_data), "sha256": compute_sha256(train_path)},
            "dev": {"file": dev_path, "count": len(dev_data), "sha256": compute_sha256(dev_path)},
            "e1a_iid_2digit": {"file": e1a_path, "count": len(e1a_data), "sha256": compute_sha256(e1a_path)},
            "e1b_iid_3digit": {"file": e1b_path, "count": len(e1b_data), "sha256": compute_sha256(e1b_path)},
            "e2_ood_2digit": {"file": e2_path, "count": len(e2_data), "sha256": compute_sha256(e2_path)},
            "e3_ood_3digit": {"file": e3_path, "count": len(e3_data), "sha256": compute_sha256(e3_path)},
            "e4_acid_4digit": {"file": e4_path, "count": len(e4_data), "sha256": compute_sha256(e4_path)},
        },
        "overlap_audit": {"status": "PASSED_ZERO_OVERLAP"},
    }

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[Saved] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
