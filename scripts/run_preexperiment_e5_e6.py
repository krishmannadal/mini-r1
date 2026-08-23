"""
Task 19B Pre-Experiment: Operator-Specific 3-Digit Evaluation

PURPOSE:
Test whether the existing compact arithmetic SFT model's 3-digit OOD failure
is operator-specific. The training data contains 362 Calculate examples with
3-digit operands, ALL in division context (e.g., "176 / 8 = 22"). If the model
performs better on 3-digit division than 3-digit addition/subtraction, this
confirms the operator x digit-length interaction hypothesis (A).

BENCHMARKS GENERATED:
E5 - 3-digit division only: N / M where N in [101,250], M in [2,12], N % M == 0
E6 - 3-digit addition/subtraction: A op B where A,B in [101,250], op in {+,-}

NO MODEL TRAINING. EVALUATION ONLY.
"""

import json
import os
import random
import re
import sys
import time

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


def verify_equations_in_trace(text: str) -> tuple[int, int]:
    """Extract equations and verify truth."""
    sym_eqs = re.findall(r"(\d+)\s*([\+\-\*\/])\s*(\d+)\s*=\s*(\d+)", text)
    valid = 0
    total = 0
    for lhs, op, rhs, res in sym_eqs:
        total += 1
        l, r, exp = int(lhs), int(rhs), int(res)
        if op == "+" and l + r == exp:
            valid += 1
        elif op == "-" and l - r == exp:
            valid += 1
        elif op == "*" and l * r == exp:
            valid += 1
        elif op == "/" and r != 0 and l % r == 0 and l // r == exp:
            valid += 1
    return valid, total


def check_repetition(tokens: list[int], n: int = 4) -> bool:
    if len(tokens) < n * 2:
        return False
    seen = set()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        if ngram in seen:
            return True
        seen.add(ngram)
    return False


@torch.inference_mode()
def batch_generate(
    model: MiniR1Transformer,
    tokenizer: Tokenizer,
    prompts: list[str],
    device: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> list[str]:
    """Batched text generation - greedy decoding."""
    encoded = [tokenizer.encode(p, add_bos=True, add_eos=False) for p in prompts]
    max_p_len = max(len(x) for x in encoded)
    pad_id = tokenizer.pad_id
    eos_id = tokenizer.eos_id

    padded_in = []
    attn_mask = []
    for seq in encoded:
        pad_len = max_p_len - len(seq)
        padded_in.append([pad_id] * pad_len + seq)
        attn_mask.append([0] * pad_len + [1] * len(seq))

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


def generate_e5_division_3digit(n_examples: int = 300, seed: int = 42) -> list[dict]:
    """
    E5: 3-digit division problems.
    Format: "Question: Calculate: N / M" where N in [101,250], M in [2,12], N % M == 0.
    This matches the training format for division with 3-digit dividends.
    """
    rng = random.Random(seed)
    examples = []
    seen = set()

    # Generate all valid (dividend, divisor) pairs in range
    valid_pairs = []
    for m in range(2, 13):  # divisors [2, 12]
        for quotient in range(2, 126):  # quotients to keep dividend in [101, 250]
            n = m * quotient
            if 101 <= n <= 250:
                valid_pairs.append((n, m, quotient))

    rng.shuffle(valid_pairs)

    for n, m, quotient in valid_pairs:
        if len(examples) >= n_examples:
            break
        key = (n, m)
        if key in seen:
            continue
        seen.add(key)

        examples.append({
            "example_id": f"e5_div3d_{len(examples):04d}",
            "level": 1,
            "prompt": f"Question: Calculate: {n} / {m}\n",
            "expression": f"{n} / {m}",
            "final_answer": str(quotient),
            "op_count": 1,
            "operations": [{
                "step": 1,
                "lhs": n,
                "op": "/",
                "rhs": m,
                "result": quotient,
                "state": quotient,
            }],
        })

    return examples


def generate_e6_addsub_3digit(n_examples: int = 600, seed: int = 42) -> list[dict]:
    """
    E6: 3-digit addition/subtraction problems.
    Format: "Question: Calculate: A + B" or "Question: Calculate: A - B"
    where A, B in [101, 250].
    The model was NEVER trained on 3-digit operands for +/-.
    """
    rng = random.Random(seed)
    examples = []
    seen = set()

    while len(examples) < n_examples:
        a = rng.randint(101, 250)
        b = rng.randint(101, 250)
        op = rng.choice(["+", "-"])

        if op == "+":
            result = a + b
        else:
            result = a - b

        key = (a, op, b)
        if key in seen:
            continue
        seen.add(key)

        examples.append({
            "example_id": f"e6_addsub3d_{len(examples):04d}",
            "level": 1,
            "prompt": f"Question: Calculate: {a} {op} {b}\n",
            "expression": f"{a} {op} {b}",
            "final_answer": str(result),
            "op_count": 1,
            "operations": [{
                "step": 1,
                "lhs": a,
                "op": op,
                "rhs": b,
                "result": result,
                "state": result,
            }],
        })

    return examples


def evaluate_dataset(model, tokenizer, dataset, device, batch_size=64, max_new_tokens=128):
    """Evaluate and return structured metrics."""
    total = len(dataset)
    correct = 0
    parsed = 0
    eos_count = 0
    rep_count = 0
    token_lengths = []
    step_valid = 0
    step_total = 0
    results = []

    for i in range(0, total, batch_size):
        batch = dataset[i : i + batch_size]
        prompts = [x["prompt"] for x in batch]
        gens = batch_generate(model, tokenizer, prompts, device, max_new_tokens=max_new_tokens)

        for item, gen in zip(batch, gens):
            extracted = extract_answer(gen)
            is_parse = extracted is not None
            is_correct = (extracted == str(item["final_answer"])) if is_parse else False

            if is_parse:
                parsed += 1
            if is_correct:
                correct += 1

            gen_toks = tokenizer.encode(gen, add_bos=False, add_eos=False)
            token_lengths.append(len(gen_toks))

            if "<answer>" in gen and "</answer>" in gen:
                eos_count += 1
            if check_repetition(gen_toks):
                rep_count += 1

            v_eq, t_eq = verify_equations_in_trace(gen)
            step_valid += v_eq
            step_total += t_eq

            results.append({
                "example_id": item.get("example_id"),
                "gold_answer": item["final_answer"],
                "extracted_answer": extracted,
                "is_correct": is_correct,
                "is_parsed": is_parse,
                "gen_length": len(gen_toks),
                "generation": gen[:300],
            })

    acc = correct / total * 100 if total > 0 else 0
    parse_rate = parsed / total * 100 if total > 0 else 0
    eos_rate = eos_count / total * 100 if total > 0 else 0
    rep_rate = rep_count / total * 100 if total > 0 else 0
    mean_toks = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    step_eq_acc = (step_valid / step_total * 100) if step_total > 0 else 0

    return {
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "parse_rate": parse_rate,
        "eos_rate": eos_rate,
        "rep_rate": rep_rate,
        "mean_tokens": mean_toks,
        "step_valid_equations": step_valid,
        "step_total_equations": step_total,
        "step_equation_accuracy": step_eq_acc,
        "sample_results": results,
    }


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("model_config", ckpt.get("config", {}))
    model_cfg = MiniR1Config.from_dict(cfg_dict)
    model = MiniR1Transformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, model_cfg


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_path = "artifacts/tokenizer/tokenizer.model"
    tokenizer = Tokenizer(tok_path)

    print("=" * 70)
    print(" TASK 19B PRE-EXPERIMENT: OPERATOR-SPECIFIC 3-DIGIT EVALUATION")
    print(f" Device: {device}")
    print("=" * 70)

    # Generate benchmarks
    print("\n--- Generating E5: 3-Digit Division (trained context) ---")
    e5_data = generate_e5_division_3digit(n_examples=300, seed=42)
    print(f"  Generated {len(e5_data)} examples")
    print(f"  Sample: {e5_data[0]['prompt'].strip()} -> {e5_data[0]['final_answer']}")

    print("\n--- Generating E6: 3-Digit Addition/Subtraction (untrained context) ---")
    e6_data = generate_e6_addsub_3digit(n_examples=600, seed=42)
    print(f"  Generated {len(e6_data)} examples")
    print(f"  Sample: {e6_data[0]['prompt'].strip()} -> {e6_data[0]['final_answer']}")

    # Save benchmarks for reproducibility
    os.makedirs("data/arithmetic_sft/task19b_benchmarks", exist_ok=True)
    with open("data/arithmetic_sft/task19b_benchmarks/e5_div_3digit.json", "w") as f:
        json.dump(e5_data, f, indent=2)
    with open("data/arithmetic_sft/task19b_benchmarks/e6_addsub_3digit.json", "w") as f:
        json.dump(e6_data, f, indent=2)
    print("  Saved to data/arithmetic_sft/task19b_benchmarks/")

    # Also split E6 by operator for finer analysis
    e6_add = [x for x in e6_data if "+" in x["expression"]]
    e6_sub = [x for x in e6_data if "-" in x["expression"]]
    print(f"  E6 breakdown: {len(e6_add)} addition, {len(e6_sub)} subtraction")

    # Evaluate R0 (compact arithmetic SFT)
    ckpt_path = "artifacts/sft/arithmetic_compact/best.pt"
    print(f"\n--- Loading Compact Arithmetic SFT (R0) from {ckpt_path} ---")
    model, _ = load_model(ckpt_path, device)

    eval_sets = [
        ("E5: 3-Digit Division (trained context)", e5_data),
        ("E6: 3-Digit Add/Sub (untrained context)", e6_data),
        ("E6-Add: 3-Digit Addition only", e6_add),
        ("E6-Sub: 3-Digit Subtraction only", e6_sub),
    ]

    all_results = {}
    for name, dset in eval_sets:
        print(f"\n  Evaluating: {name} (N={len(dset)})")
        t0 = time.perf_counter()
        res = evaluate_dataset(model, tokenizer, dset, device, batch_size=64, max_new_tokens=128)
        elapsed = time.perf_counter() - t0
        print(f"    Accuracy: {res['accuracy']:.2f}%")
        print(f"    Parse rate: {res['parse_rate']:.1f}%")
        print(f"    EOS rate: {res['eos_rate']:.1f}%")
        print(f"    Repetition rate: {res['rep_rate']:.1f}%")
        print(f"    Mean tokens: {res['mean_tokens']:.1f}")
        print(f"    Step equation accuracy: {res['step_equation_accuracy']:.1f}%")
        print(f"    Time: {elapsed:.1f}s")
        all_results[name] = res

    # Print sample generations for inspection
    for name, res in all_results.items():
        print(f"\n{'=' * 60}")
        print(f"  SAMPLE GENERATIONS: {name}")
        print(f"{'=' * 60}")
        for sample in res["sample_results"][:5]:
            print(f"  ID: {sample['example_id']}")
            print(f"  Gold: {sample['gold_answer']}")
            print(f"  Extracted: {sample['extracted_answer']}")
            print(f"  Correct: {sample['is_correct']}")
            print(f"  Generation: {sample['generation'][:200]}")
            print()

    # Summary comparison
    e5_acc = all_results["E5: 3-Digit Division (trained context)"]["accuracy"]
    e6_acc = all_results["E6: 3-Digit Add/Sub (untrained context)"]["accuracy"]
    delta = e5_acc - e6_acc

    print("\n" + "=" * 70)
    print(" PRE-EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    print(f"  E5 (3-digit div, trained context):    {e5_acc:.2f}%")
    print(f"  E6 (3-digit +/-, untrained context):  {e6_acc:.2f}%")
    print(f"  Delta (E5 - E6):                      {delta:+.2f}%")
    print()
    if delta > 10:
        print("  INTERPRETATION: E5 >> E6 -> 3-digit failure IS operator-specific.")
        print("  The model learned 3-digit arithmetic in the division context but")
        print("  cannot transfer to addition/subtraction. Hypothesis A CONFIRMED.")
    elif delta > 3:
        print("  INTERPRETATION: E5 > E6 -> Moderate operator-specificity detected.")
        print("  Mixed evidence for Hypothesis A. Proceed with R1.")
    else:
        print("  INTERPRETATION: E5 ~ E6 -> 3-digit failure is NOT operator-specific.")
        print("  The model fails equally on 3-digit arithmetic regardless of operator.")
        print("  Hypothesis A (operator x digit-length) NOT confirmed.")
        print("  Hypothesis B (value memorization) or general 3-digit failure more likely.")
    print("=" * 70)

    # Save full results
    compact_results = {}
    for name, res in all_results.items():
        compact_results[name] = {k: v for k, v in res.items() if k != "sample_results"}
        compact_results[name]["first_10_generations"] = [
            {k: v for k, v in s.items()} for s in res["sample_results"][:10]
        ]

    with open("reports/task19b_preexperiment_results.json", "w") as f:
        json.dump(compact_results, f, indent=2)
    print("\nFull results saved to reports/task19b_preexperiment_results.json")


if __name__ == "__main__":
    main()
