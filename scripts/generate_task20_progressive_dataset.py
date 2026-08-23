"""
generate_task20_progressive_dataset.py - Task 20: Progressive Numeric-Regime Dataset Generator.

Curriculum:
- Train: 12,000 examples (4,000 from 2D [50,99], 4,000 from 3D [101,199], 4,000 from 4D [1001,1999])
- Dev: 1,200 examples (400 per regime)
- 7 Zero-Overlap Evaluation Sets (500 examples each):
  * E1: IID 2D [50,99]
  * E2: OOD 2D [2,49]
  * E3: IID 3D [101,199]
  * E4: OOD 3D [200,299]
  * E5: IID 4D [1001,1999]
  * E6: OOD 4D [2000,2999]
  * E7: Acid Test 5D [10001,19999]
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


def format_train_example(example_id: str, a: int, b: int, regime: str) -> dict[str, Any]:
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
            "regime": regime,
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


def sample_large_pairs(lo: int, hi: int, n: int, rng: random.Random) -> list[tuple[int, int]]:
    """Sample unique (a, b) with lo <= b <= a <= hi."""
    seen = set()
    pairs = []
    while len(pairs) < n:
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        if a < b:
            a, b = b, a
        key = (a, b)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def main():
    seed = 42
    rng = random.Random(seed)
    out_dir = "data/progressive_subtraction_sft"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print(" TASK 20: PROGRESSIVE NUMERIC-REGIME DATASET GENERATOR")
    print(f" Seed: {seed} | Output Dir: {out_dir}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. 2D Regime: [50, 99] (1,275 total pairs)
    # -------------------------------------------------------------------------
    all_2d_pairs = enumerate_pairs(50, 99)
    rng.shuffle(all_2d_pairs)
    train_2d_pool = all_2d_pairs[:1000]
    holdout_2d_pool = all_2d_pairs[1000:]
    train_2d_set = set(train_2d_pool)
    holdout_2d_set = set(holdout_2d_pool)
    assert len(train_2d_set & holdout_2d_set) == 0

    # -------------------------------------------------------------------------
    # 2. 3D Regime: [101, 199] (4,950 total pairs)
    # -------------------------------------------------------------------------
    all_3d_pairs = enumerate_pairs(101, 199)
    rng.shuffle(all_3d_pairs)
    train_3d_pool = all_3d_pairs[:4000]
    holdout_3d_pool = all_3d_pairs[4000:]
    train_3d_set = set(train_3d_pool)
    holdout_3d_set = set(holdout_3d_pool)
    assert len(train_3d_set & holdout_3d_set) == 0

    # -------------------------------------------------------------------------
    # 3. 4D Regime: [1001, 1999] (~500,000 total pairs)
    # -------------------------------------------------------------------------
    # Sample 4,000 for train pool, 1,000 for holdout pool
    all_4d_pool = sample_large_pairs(1001, 1999, 5000, rng)
    train_4d_pool = all_4d_pool[:4000]
    holdout_4d_pool = all_4d_pool[4000:]
    train_4d_set = set(train_4d_pool)
    holdout_4d_set = set(holdout_4d_pool)
    assert len(train_4d_set & holdout_4d_set) == 0

    print(f"2D Pool: {len(train_2d_pool)} train / {len(holdout_2d_pool)} holdout")
    print(f"3D Pool: {len(train_3d_pool)} train / {len(holdout_3d_pool)} holdout")
    print(f"4D Pool: {len(train_4d_pool)} train / {len(holdout_4d_pool)} holdout")

    # -------------------------------------------------------------------------
    # 4. Generate Training Set (12,000 = 4,000 2D + 4,000 3D + 4,000 4D)
    # -------------------------------------------------------------------------
    train_items = []
    for i in range(4000):
        train_items.append((train_2d_pool[i % len(train_2d_pool)], "2D"))
    for i in range(4000):
        train_items.append((train_3d_pool[i % len(train_3d_pool)], "3D"))
    for i in range(4000):
        train_items.append((train_4d_pool[i % len(train_4d_pool)], "4D"))

    rng.shuffle(train_items)

    train_data = [
        format_train_example(f"prog_train_{i+1:05d}", pair[0], pair[1], regime)
        for i, (pair, regime) in enumerate(train_items)
    ]

    train_path = os.path.join(out_dir, "train.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")
    print(f"\n[Generated] {train_path} ({len(train_data)} items)")

    # -------------------------------------------------------------------------
    # 5. Generate Dev Set (1,200 = 400 2D + 400 3D + 400 4D)
    # -------------------------------------------------------------------------
    dev_items = []
    for _ in range(400):
        dev_items.append((rng.choice(train_2d_pool), "2D"))
    for _ in range(400):
        dev_items.append((rng.choice(train_3d_pool), "3D"))
    for _ in range(400):
        dev_items.append((rng.choice(train_4d_pool), "4D"))

    rng.shuffle(dev_items)

    dev_data = [
        format_train_example(f"prog_dev_{i+1:04d}", pair[0], pair[1], regime)
        for i, (pair, regime) in enumerate(dev_items)
    ]

    dev_path = os.path.join(out_dir, "dev.jsonl")
    with open(dev_path, "w", encoding="utf-8") as f:
        for ex in dev_data:
            f.write(json.dumps(ex) + "\n")
    print(f"[Generated] {dev_path} ({len(dev_data)} items)")

    # -------------------------------------------------------------------------
    # 6. Generate Seven Evaluation Benchmarks (500 items each)
    # -------------------------------------------------------------------------

    # E1: IID 2D [50, 99] (500 items from holdout_2d_pool)
    e1_items = list(holdout_2d_pool)
    while len(e1_items) < 500:
        e1_items.append(rng.choice(holdout_2d_pool))
    rng.shuffle(e1_items)
    e1_items = e1_items[:500]
    e1_data = [format_eval_example(f"e1_iid2d_{i+1:04d}", a, b) for i, (a, b) in enumerate(e1_items)]
    e1_path = os.path.join(out_dir, "eval_e1_iid_2digit.json")
    with open(e1_path, "w", encoding="utf-8") as f:
        json.dump(e1_data, f, indent=2)
    print(f"[Generated] {e1_path} ({len(e1_data)} items)")

    # E2: OOD 2D [2, 49] (500 unique items)
    all_e2 = enumerate_pairs(2, 49)
    rng.shuffle(all_e2)
    e2_items = all_e2[:500]
    e2_data = [format_eval_example(f"e2_ood2d_{i+1:04d}", a, b) for i, (a, b) in enumerate(e2_items)]
    e2_path = os.path.join(out_dir, "eval_e2_ood_2digit.json")
    with open(e2_path, "w", encoding="utf-8") as f:
        json.dump(e2_data, f, indent=2)
    print(f"[Generated] {e2_path} ({len(e2_data)} items)")

    # E3: IID 3D [101, 199] (500 unique items from holdout_3d_pool)
    e3_items = list(holdout_3d_pool[:500])
    e3_data = [format_eval_example(f"e3_iid3d_{i+1:04d}", a, b) for i, (a, b) in enumerate(e3_items)]
    e3_path = os.path.join(out_dir, "eval_e3_iid_3digit.json")
    with open(e3_path, "w", encoding="utf-8") as f:
        json.dump(e3_data, f, indent=2)
    print(f"[Generated] {e3_path} ({len(e3_data)} items)")

    # E4: OOD 3D [200, 299] (500 unique items)
    all_e4 = enumerate_pairs(200, 299)
    rng.shuffle(all_e4)
    e4_items = all_e4[:500]
    e4_data = [format_eval_example(f"e4_ood3d_{i+1:04d}", a, b) for i, (a, b) in enumerate(e4_items)]
    e4_path = os.path.join(out_dir, "eval_e4_ood_3digit.json")
    with open(e4_path, "w", encoding="utf-8") as f:
        json.dump(e4_data, f, indent=2)
    print(f"[Generated] {e4_path} ({len(e4_data)} items)")

    # E5: IID 4D [1001, 1999] (500 unique items from holdout_4d_pool)
    e5_items = list(holdout_4d_pool[:500])
    e5_data = [format_eval_example(f"e5_iid4d_{i+1:04d}", a, b) for i, (a, b) in enumerate(e5_items)]
    e5_path = os.path.join(out_dir, "eval_e5_iid_4digit.json")
    with open(e5_path, "w", encoding="utf-8") as f:
        json.dump(e5_data, f, indent=2)
    print(f"[Generated] {e5_path} ({len(e5_data)} items)")

    # E6: OOD 4D [2000, 2999] (500 unique items)
    e6_items = sample_large_pairs(2000, 2999, 500, rng)
    e6_data = [format_eval_example(f"e6_ood4d_{i+1:04d}", a, b) for i, (a, b) in enumerate(e6_items)]
    e6_path = os.path.join(out_dir, "eval_e6_ood_4digit.json")
    with open(e6_path, "w", encoding="utf-8") as f:
        json.dump(e6_data, f, indent=2)
    print(f"[Generated] {e6_path} ({len(e6_data)} items)")

    # E7: Acid Test 5D [10001, 19999] (500 unique items)
    e7_items = sample_large_pairs(10001, 19999, 500, rng)
    e7_data = [format_eval_example(f"e7_acid5d_{i+1:04d}", a, b) for i, (a, b) in enumerate(e7_items)]
    e7_path = os.path.join(out_dir, "eval_e7_acid_5digit.json")
    with open(e7_path, "w", encoding="utf-8") as f:
        json.dump(e7_data, f, indent=2)
    print(f"[Generated] {e7_path} ({len(e7_data)} items)")

    # -------------------------------------------------------------------------
    # 7. Rigorous Overlap Audit
    # -------------------------------------------------------------------------
    print("\n--- Running Rigorous Overlap Audit ---")
    train_pairs_set = set((ex["metadata"]["lhs"], ex["metadata"]["rhs"]) for ex in train_data)
    dev_pairs_set = set((ex["metadata"]["lhs"], ex["metadata"]["rhs"]) for ex in dev_data)

    eval_sets = {
        "E1": set((ex["lhs"], ex["rhs"]) for ex in e1_data),
        "E2": set((ex["lhs"], ex["rhs"]) for ex in e2_data),
        "E3": set((ex["lhs"], ex["rhs"]) for ex in e3_data),
        "E4": set((ex["lhs"], ex["rhs"]) for ex in e4_data),
        "E5": set((ex["lhs"], ex["rhs"]) for ex in e5_data),
        "E6": set((ex["lhs"], ex["rhs"]) for ex in e6_data),
        "E7": set((ex["lhs"], ex["rhs"]) for ex in e7_data),
    }

    all_pass = True
    overlap_results = {}
    for name, eset in eval_sets.items():
        ov_train = len(eset & train_pairs_set)
        ov_dev = len(eset & dev_pairs_set)
        status = "PASS" if ov_train == 0 and ov_dev == 0 else "FAIL"
        if status == "FAIL":
            all_pass = False
        overlap_results[name] = {"N": len(eset), "train_overlap": ov_train, "dev_overlap": ov_dev, "status": status}
        print(f"  {name} (N={len(eset)}): train_overlap={ov_train}, dev_overlap={ov_dev} [{status}]")

    assert all_pass, "OVERLAP AUDIT FAILED!"
    print(">>> OVERLAP AUDIT PASSED: EXACT 0 OVERLAP CONFIRMED ACROSS ALL 7 SETS! <<<")

    # -------------------------------------------------------------------------
    # 8. Digit Length Summary
    # -------------------------------------------------------------------------
    def digit_counts(dataset: list[dict[str, Any]]) -> dict[int, int]:
        c = Counter(ex.get("ans_digits", ex.get("metadata", {}).get("ans_digits")) for ex in dataset)
        return dict(sorted(c.items()))

    print("\n--- Answer Digit-Length Distributions ---")
    print(f"  Train: {digit_counts(train_data)}")
    for name, path in [("E1", e1_path), ("E2", e2_path), ("E3", e3_path), ("E4", e4_path),
                       ("E5", e5_path), ("E6", e6_path), ("E7", e7_path)]:
        with open(path) as f:
            d = json.load(f)
        print(f"  {name}: {digit_counts(d)}")

    # -------------------------------------------------------------------------
    # 9. Save Manifest
    # -------------------------------------------------------------------------
    manifest = {
        "dataset_name": "Task 20 Progressive Numeric-Regime Subtraction SFT",
        "seed": seed,
        "curriculum_composition": {
            "2D_regime": {"range": "[50, 99]", "count": 4000, "unique_pairs": len(train_2d_set)},
            "3D_regime": {"range": "[101, 199]", "count": 4000, "unique_pairs": len(train_3d_set)},
            "4D_regime": {"range": "[1001, 1999]", "count": 4000, "unique_pairs": len(train_4d_set)},
            "total_train": len(train_data),
            "total_dev": len(dev_data),
        },
        "splits": {
            "train": {"file": train_path, "count": len(train_data), "sha256": compute_sha256(train_path)},
            "dev": {"file": dev_path, "count": len(dev_data), "sha256": compute_sha256(dev_path)},
            "e1_iid_2digit": {"file": e1_path, "count": len(e1_data), "sha256": compute_sha256(e1_path)},
            "e2_ood_2digit": {"file": e2_path, "count": len(e2_data), "sha256": compute_sha256(e2_path)},
            "e3_iid_3digit": {"file": e3_path, "count": len(e3_data), "sha256": compute_sha256(e3_path)},
            "e4_ood_3digit": {"file": e4_path, "count": len(e4_data), "sha256": compute_sha256(e4_path)},
            "e5_iid_4digit": {"file": e5_path, "count": len(e5_data), "sha256": compute_sha256(e5_path)},
            "e6_ood_4digit": {"file": e6_path, "count": len(e6_data), "sha256": compute_sha256(e6_path)},
            "e7_acid_5digit": {"file": e7_path, "count": len(e7_data), "sha256": compute_sha256(e7_path)},
        },
        "overlap_audit": {
            "status": "PASSED_ZERO_OVERLAP",
            "per_benchmark": overlap_results,
        },
    }

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[Saved] Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
