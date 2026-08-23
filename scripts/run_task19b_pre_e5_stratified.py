"""
Task 19B-PRE Addendum: E5-Stratified Overlap Analysis

Zero-cost diagnostic: re-run E5 evaluation on the frozen Compact Arithmetic SFT
checkpoint, then stratify accuracy by:
1. Exact training match vs non-match
2. Quotient range ([2,25] trained vs [26+] OOD)
3. Cross of both

This definitively confirms or refutes the memorization hypothesis.
"""

import json
import os
import random
import re
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import MiniR1Config, MiniR1Transformer
from src.tokenizer import Tokenizer


def extract_answer(text: str) -> str | None:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        raw = match.group(1).strip()
        num_m = re.search(r"-?\d+", raw)
        return num_m.group(0) if num_m else raw
    return None


@torch.inference_mode()
def batch_generate(model, tokenizer, prompts, device, max_new_tokens=128):
    """Greedy batched generation."""
    encoded = [tokenizer.encode(p, add_bos=True, add_eos=False) for p in prompts]
    max_p_len = max(len(x) for x in encoded)
    pad_id = tokenizer.pad_id
    eos_id = tokenizer.eos_id

    padded_in = []
    for seq in encoded:
        pad_len = max_p_len - len(seq)
        padded_in.append([pad_id] * pad_len + seq)

    input_tensor = torch.tensor(padded_in, dtype=torch.long, device=device)
    batch_size = input_tensor.size(0)
    generated = [[] for _ in range(batch_size)]
    finished = [False] * batch_size

    for _ in range(max_new_tokens):
        if all(finished):
            break
        with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16):
            logits = model(input_tensor)
            next_logits = logits[:, -1, :].clone()
        next_tokens = torch.argmax(next_logits, dim=-1)
        for b in range(batch_size):
            if not finished[b]:
                tok = next_tokens[b].item()
                if tok == eos_id:
                    finished[b] = True
                else:
                    generated[b].append(tok)
        input_tensor = torch.cat([input_tensor, next_tokens.unsqueeze(1)], dim=1)

    return [tokenizer.decode(g) for g in generated]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer("artifacts/tokenizer/tokenizer.model")

    print("=" * 80)
    print(" TASK 19B-PRE ADDENDUM: E5 STRATIFIED OVERLAP ANALYSIS")
    print("=" * 80)

    # 1. Build training overlap set
    print("\n[1/4] Building training overlap set...")
    train_div_pairs = set()
    with open("data/arithmetic_sft/compact_train.jsonl", "r") as f:
        for line in f:
            rec = json.loads(line)
            prompt = rec["prompt"]
            if "Calculate:" in prompt and "/" in prompt:
                m = re.search(r"(\d+)\s*/\s*(\d+)", prompt)
                if m:
                    train_div_pairs.add((int(m.group(1)), int(m.group(2))))
            # Also check multi-step problems that contain division
            comp = rec["completion"]
            for eq in re.findall(r"(\d+)\s*/\s*(\d+)\s*=\s*(\d+)", comp):
                train_div_pairs.add((int(eq[0]), int(eq[1])))

    print(f"  Total unique (dividend, divisor) pairs in training: {len(train_div_pairs)}")

    # 2. Reconstruct E5 dataset (deterministic, same seed as evaluation)
    print("\n[2/4] Reconstructing E5 dataset (seed=42)...")
    rng = random.Random(42)
    valid_pairs = []
    for divisor in range(2, 13):
        for quotient in range(2, 126):
            dividend = divisor * quotient
            if 101 <= dividend <= 250:
                valid_pairs.append((dividend, divisor, quotient))

    rng.shuffle(valid_pairs)
    e5_data = []
    seen = set()
    for dividend, divisor, quotient in valid_pairs:
        if len(e5_data) >= 500:
            break
        key = (dividend, divisor)
        if key in seen:
            continue
        seen.add(key)

        in_train = key in train_div_pairs
        q_range = "[2,25]" if quotient <= 25 else "[26+]"

        e5_data.append({
            "dividend": dividend,
            "divisor": divisor,
            "quotient": quotient,
            "in_training": in_train,
            "q_range": q_range,
            "prompt": f"Question: Calculate: {dividend} / {divisor}\n",
            "gold_answer": str(quotient),
        })

    n_train_match = sum(1 for x in e5_data if x["in_training"])
    n_non_match = sum(1 for x in e5_data if not x["in_training"])
    print(f"  E5 total: {len(e5_data)}")
    print(f"  Exact training matches: {n_train_match}")
    print(f"  Non-training examples: {n_non_match}")

    # 3. Load model and evaluate
    print("\n[3/4] Loading model and running greedy evaluation...")
    ckpt = torch.load("artifacts/sft/arithmetic_compact/best.pt", map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("model_config", ckpt.get("config", {}))
    from src.model import MiniR1Config, MiniR1Transformer
    model_cfg = MiniR1Config.from_dict(cfg_dict)
    model = MiniR1Transformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Run inference in batches
    batch_size = 64
    all_results = []
    t0 = time.perf_counter()

    for i in range(0, len(e5_data), batch_size):
        batch = e5_data[i:i + batch_size]
        prompts = [x["prompt"] for x in batch]
        gens = batch_generate(model, tokenizer, prompts, device, max_new_tokens=128)

        for item, gen in zip(batch, gens):
            extracted = extract_answer(gen)
            is_correct = (extracted == item["gold_answer"]) if extracted else False

            all_results.append({
                "dividend": item["dividend"],
                "divisor": item["divisor"],
                "quotient": item["quotient"],
                "in_training": item["in_training"],
                "q_range": item["q_range"],
                "gold_answer": item["gold_answer"],
                "extracted_answer": extracted,
                "is_correct": is_correct,
                "generation": gen.strip(),
            })

    elapsed = time.perf_counter() - t0
    print(f"  Evaluation complete: {elapsed:.1f}s")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Stratified Analysis
    print("\n[4/4] Computing Stratified Results...")

    # Overall
    total = len(all_results)
    correct = sum(1 for r in all_results if r["is_correct"])
    print(f"\n  OVERALL: {correct}/{total} = {correct/total*100:.2f}%")

    # By training overlap
    strata = {
        "Training Match": [r for r in all_results if r["in_training"]],
        "Non-Training": [r for r in all_results if not r["in_training"]],
    }

    print(f"\n  {'Stratum':<25} {'N':>5} {'Correct':>8} {'Accuracy':>10}")
    print(f"  {'-'*50}")
    for label, items in strata.items():
        n = len(items)
        c = sum(1 for r in items if r["is_correct"])
        acc = c / n * 100 if n > 0 else 0
        print(f"  {label:<25} {n:>5} {c:>8} {acc:>9.2f}%")

    # By quotient range
    q_strata = {
        "Quotient [2,25]": [r for r in all_results if r["q_range"] == "[2,25]"],
        "Quotient [26+]": [r for r in all_results if r["q_range"] == "[26+]"],
    }

    print(f"\n  {'Stratum':<25} {'N':>5} {'Correct':>8} {'Accuracy':>10}")
    print(f"  {'-'*50}")
    for label, items in q_strata.items():
        n = len(items)
        c = sum(1 for r in items if r["is_correct"])
        acc = c / n * 100 if n > 0 else 0
        print(f"  {label:<25} {n:>5} {c:>8} {acc:>9.2f}%")

    # Cross: training × quotient range
    cross_strata = {
        "Train + Q[2,25]": [r for r in all_results if r["in_training"] and r["q_range"] == "[2,25]"],
        "Train + Q[26+]": [r for r in all_results if r["in_training"] and r["q_range"] == "[26+]"],
        "NonTrain + Q[2,25]": [r for r in all_results if not r["in_training"] and r["q_range"] == "[2,25]"],
        "NonTrain + Q[26+]": [r for r in all_results if not r["in_training"] and r["q_range"] == "[26+]"],
    }

    print(f"\n  {'Stratum':<25} {'N':>5} {'Correct':>8} {'Accuracy':>10}")
    print(f"  {'-'*50}")
    for label, items in cross_strata.items():
        n = len(items)
        c = sum(1 for r in items if r["is_correct"])
        acc = c / n * 100 if n > 0 else 0
        print(f"  {label:<25} {n:>5} {c:>8} {acc:>9.2f}%")

    # Model output distribution analysis on non-training examples
    non_train_outputs = []
    for r in all_results:
        if not r["in_training"] and r["extracted_answer"] is not None:
            try:
                non_train_outputs.append(int(r["extracted_answer"]))
            except ValueError:
                pass

    if non_train_outputs:
        arr = np.array(non_train_outputs)
        print(f"\n  NON-TRAINING MODEL OUTPUT DISTRIBUTION:")
        print(f"    N outputs parsed: {len(arr)}")
        print(f"    Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean():.1f}, Median: {np.median(arr):.0f}")
        print(f"    Outputs in [2,25]: {np.sum((arr >= 2) & (arr <= 25))} ({np.sum((arr >= 2) & (arr <= 25))/len(arr)*100:.1f}%)")
        print(f"    Outputs in [26+]: {np.sum(arr >= 26)} ({np.sum(arr >= 26)/len(arr)*100:.1f}%)")

    # Show specific examples: non-training correct (if any)
    non_train_correct = [r for r in all_results if not r["in_training"] and r["is_correct"]]
    print(f"\n  NON-TRAINING CORRECT EXAMPLES: {len(non_train_correct)}")
    for r in non_train_correct[:20]:
        print(f"    {r['dividend']} / {r['divisor']} = {r['quotient']} (gold) | model: {r['extracted_answer']} | gen: {r['generation'][:80]}")

    # Show specific examples: training match incorrect (if any)
    train_incorrect = [r for r in all_results if r["in_training"] and not r["is_correct"]]
    print(f"\n  TRAINING MATCH BUT INCORRECT: {len(train_incorrect)}")
    for r in train_incorrect[:20]:
        print(f"    {r['dividend']} / {r['divisor']} = {r['quotient']} (gold) | model: {r['extracted_answer']} | gen: {r['generation'][:80]}")

    # Summary verdict
    train_match_items = strata["Training Match"]
    non_train_items = strata["Non-Training"]
    train_correct = sum(1 for r in train_match_items if r["is_correct"])
    non_train_correct_n = sum(1 for r in non_train_items if r["is_correct"])

    print(f"\n{'=' * 80}")
    print(f" VERDICT")
    print(f"{'=' * 80}")
    print(f"  Training matches: {len(train_match_items)} examples, {train_correct} correct ({train_correct/len(train_match_items)*100:.1f}% recall)")
    print(f"  Non-training:     {len(non_train_items)} examples, {non_train_correct_n} correct ({non_train_correct_n/len(non_train_items)*100:.2f}% accuracy)")
    print(f"")
    if non_train_correct_n / len(non_train_items) < 0.03:
        print(f"  CONFIRMED: E5 accuracy ({correct/total*100:.2f}%) is EXPLAINED BY MEMORIZATION.")
        print(f"  The model recalls {train_correct/len(train_match_items)*100:.1f}% of training examples")
        print(f"  and achieves only {non_train_correct_n/len(non_train_items)*100:.2f}% on non-training examples.")
        print(f"  There is NO evidence of 3-digit division generalization.")
    else:
        print(f"  PARTIAL: Some non-training accuracy detected ({non_train_correct_n/len(non_train_items)*100:.2f}%).")
        print(f"  Investigate whether these are near-training interpolations.")

    # Save results
    export = {
        "metadata": {
            "task": "Task 19B-PRE Addendum: E5 Stratified Overlap Analysis",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "overall": {"total": total, "correct": correct, "accuracy": correct / total * 100},
        "by_training_overlap": {
            "training_match": {"n": len(train_match_items), "correct": train_correct, "accuracy": train_correct / len(train_match_items) * 100},
            "non_training": {"n": len(non_train_items), "correct": non_train_correct_n, "accuracy": non_train_correct_n / len(non_train_items) * 100},
        },
        "by_quotient_range": {},
        "cross_stratification": {},
        "non_training_correct_examples": [
            {k: v for k, v in r.items() if k != "generation"} for r in non_train_correct
        ],
        "training_incorrect_examples": [
            {k: v for k, v in r.items() if k != "generation"} for r in train_incorrect
        ],
    }

    for label, items in q_strata.items():
        n = len(items)
        c = sum(1 for r in items if r["is_correct"])
        export["by_quotient_range"][label] = {"n": n, "correct": c, "accuracy": c / n * 100 if n > 0 else 0}

    for label, items in cross_strata.items():
        n = len(items)
        c = sum(1 for r in items if r["is_correct"])
        export["cross_stratification"][label] = {"n": n, "correct": c, "accuracy": c / n * 100 if n > 0 else 0}

    with open("reports/task19b_pre_e5_stratified.json", "w") as f:
        json.dump(export, f, indent=2)
    print(f"\n[Saved] reports/task19b_pre_e5_stratified.json")


if __name__ == "__main__":
    main()
