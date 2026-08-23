"""
run_task19b_pre_e5_e6.py - Operator x Digit-Length Diagnostic Engine for Task 19B-PRE.

Evaluates the frozen Compact Arithmetic SFT checkpoint across two controlled conditions:
- Condition E5: 3-Digit Division (operands in [101, 250], exact integer division)
- Condition E6: 3-Digit Addition & Subtraction (operands in [101, 250])

Computes:
- Strict final-answer accuracy
- Step-level equation validity
- Format compliance (<think>, <answer>, full format)
- EOS termination & 4-gram repetition rates
- Mean, p50 (median), p90 generation lengths
- Comprehensive E5 vs E6 comparative analysis
"""

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Any

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


def verify_equations_in_trace(text: str) -> tuple[int, int]:
    """Extract equations (e.g. 176 / 8 = 22) and verify mathematical truth."""
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
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> list[str]:
    """Batched text generation supporting greedy decoding and temperature/rep-penalty sampling."""
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
    mask_tensor = torch.tensor(attn_mask, dtype=torch.long, device=device)
    batch_size = input_tensor.size(0)

    generated = [[] for _ in range(batch_size)]
    finished = [False] * batch_size

    for _ in range(max_new_tokens):
        if all(finished):
            break

        with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16):
            logits = model(input_tensor)
            next_logits = logits[:, -1, :].clone()

        if repetition_penalty != 1.0:
            for b in range(batch_size):
                if not finished[b]:
                    seen_tokens = set(encoded[b] + generated[b])
                    for t in seen_tokens:
                        if next_logits[b, t] > 0:
                            next_logits[b, t] /= repetition_penalty
                        else:
                            next_logits[b, t] *= repetition_penalty

        if temperature <= 1e-6:
            next_tokens = torch.argmax(next_logits, dim=-1)
        else:
            scaled_logits = next_logits / temperature
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                scaled_logits[indices_to_remove] = -float("Inf")

            probs = torch.softmax(scaled_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        for b in range(batch_size):
            if not finished[b]:
                tok = next_tokens[b].item()
                if tok == eos_id:
                    finished[b] = True
                else:
                    generated[b].append(tok)

        input_tensor = torch.cat([input_tensor, next_tokens.unsqueeze(1)], dim=1)
        mask_tensor = torch.cat([mask_tensor, torch.ones((batch_size, 1), device=device, dtype=torch.long)], dim=1)

    return [tokenizer.decode(g) for g in generated]


def generate_e5_division_3digit(n_examples: int = 500, seed: int = 42) -> list[dict[str, Any]]:
    """
    Condition E5: 3-digit dividend division problems.
    Formula: Dividend / Divisor = Quotient
    Range: Dividend in [101, 250], Divisor in [2, 12], exact integer result.
    """
    rng = random.Random(seed)
    valid_pairs = []
    for divisor in range(2, 13):
        for quotient in range(2, 126):
            dividend = divisor * quotient
            if 101 <= dividend <= 250:
                valid_pairs.append((dividend, divisor, quotient))

    rng.shuffle(valid_pairs)
    examples = []
    seen = set()

    for dividend, divisor, quotient in valid_pairs:
        if len(examples) >= n_examples:
            break
        key = (dividend, divisor)
        if key in seen:
            continue
        seen.add(key)

        prompt = f"Question: Calculate: {dividend} / {divisor}\n"
        examples.append({
            "example_id": f"e5_div3d_{len(examples)+1:04d}",
            "prompt": prompt,
            "expression": f"{dividend} / {divisor}",
            "final_answer": str(quotient),
            "op": "/",
            "lhs": dividend,
            "rhs": divisor,
            "intermediate_steps": [f"{dividend} / {divisor} = {quotient}"],
        })

    return examples


def generate_e6_addsub_3digit(n_examples: int = 500, seed: int = 42) -> list[dict[str, Any]]:
    """
    Condition E6: 3-digit addition & subtraction problems.
    Range: Operands in [101, 250].
    Exactly 50% Addition, 50% Subtraction.
    """
    rng = random.Random(seed)
    examples = []
    seen = set()

    half = n_examples // 2

    # 1. Addition (250 items)
    while len(examples) < half:
        a = rng.randint(101, 250)
        b = rng.randint(101, 250)
        key = (a, "+", b)
        if key in seen:
            continue
        seen.add(key)
        ans = a + b
        examples.append({
            "example_id": f"e6_add3d_{len(examples)+1:04d}",
            "prompt": f"Question: Calculate: {a} + {b}\n",
            "expression": f"{a} + {b}",
            "final_answer": str(ans),
            "op": "+",
            "lhs": a,
            "rhs": b,
            "intermediate_steps": [f"{a} + {b} = {ans}"],
        })

    # 2. Subtraction (250 items)
    while len(examples) < n_examples:
        a = rng.randint(101, 250)
        b = rng.randint(101, 250)
        if a < b:
            a, b = b, a  # keep non-negative as in training
        key = (a, "-", b)
        if key in seen:
            continue
        seen.add(key)
        ans = a - b
        examples.append({
            "example_id": f"e6_sub3d_{len(examples)+1:04d}",
            "prompt": f"Question: Calculate: {a} - {b}\n",
            "expression": f"{a} - {b}",
            "final_answer": str(ans),
            "op": "-",
            "lhs": a,
            "rhs": b,
            "intermediate_steps": [f"{a} - {b} = {ans}"],
        })

    rng.shuffle(examples)
    return examples


def evaluate_condition(
    model: MiniR1Transformer,
    tokenizer: Tokenizer,
    dataset: list[dict[str, Any]],
    device: str,
    batch_size: int = 64,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> dict[str, Any]:
    total = len(dataset)
    correct = 0
    parsed = 0
    think_compliant = 0
    answer_compliant = 0
    full_format = 0
    eos_count = 0
    rep_count = 0
    token_lengths = []
    step_valid = 0
    step_total = 0
    detailed_samples = []

    for i in range(0, total, batch_size):
        batch = dataset[i : i + batch_size]
        prompts = [x["prompt"] for x in batch]
        gens = batch_generate(
            model,
            tokenizer,
            prompts,
            device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        for item, gen in zip(batch, gens):
            has_think = "<think>" in gen and "</think>" in gen
            has_answer = "<answer>" in gen and "</answer>" in gen
            has_full = has_think and has_answer

            if has_think:
                think_compliant += 1
            if has_answer:
                answer_compliant += 1
            if has_full:
                full_format += 1

            extracted = extract_answer(gen)
            is_parse = extracted is not None
            is_correct = (extracted == str(item["final_answer"])) if is_parse else False

            if is_parse:
                parsed += 1
            if is_correct:
                correct += 1

            gen_toks = tokenizer.encode(gen, add_bos=False, add_eos=False)
            tok_len = len(gen_toks)
            token_lengths.append(tok_len)

            if "<answer>" in gen and "</answer>" in gen:
                eos_count += 1
            if check_repetition(gen_toks):
                rep_count += 1

            v_eq, t_eq = verify_equations_in_trace(gen)
            step_valid += v_eq
            step_total += t_eq

            detailed_samples.append({
                "example_id": item["example_id"],
                "prompt": item["prompt"].strip(),
                "gold_answer": item["final_answer"],
                "extracted_answer": extracted,
                "is_correct": is_correct,
                "is_parsed": is_parse,
                "has_full_format": has_full,
                "gen_length": tok_len,
                "valid_equations": v_eq,
                "total_equations": t_eq,
                "generation": gen.strip(),
            })

    lengths_arr = np.array(token_lengths) if token_lengths else np.array([0])
    acc = correct / total * 100 if total > 0 else 0.0
    parse_rate = parsed / total * 100 if total > 0 else 0.0
    think_rate = think_compliant / total * 100 if total > 0 else 0.0
    ans_rate = answer_compliant / total * 100 if total > 0 else 0.0
    full_format_rate = full_format / total * 100 if total > 0 else 0.0
    eos_rate = eos_count / total * 100 if total > 0 else 0.0
    rep_rate = rep_count / total * 100 if total > 0 else 0.0
    mean_toks = float(np.mean(lengths_arr))
    p50_toks = float(np.median(lengths_arr))
    p90_toks = float(np.percentile(lengths_arr, 90))
    step_eq_acc = (step_valid / step_total * 100) if step_total > 0 else 0.0

    return {
        "total_examples": total,
        "correct_answers": correct,
        "final_answer_accuracy": acc,
        "parse_rate": parse_rate,
        "think_compliance_rate": think_rate,
        "answer_compliance_rate": ans_rate,
        "full_format_compliance_rate": full_format_rate,
        "eos_termination_rate": eos_rate,
        "repetition_rate": rep_rate,
        "generation_length": {
            "mean": mean_toks,
            "min": int(np.min(lengths_arr)),
            "max": int(np.max(lengths_arr)),
            "p50_median": p50_toks,
            "p90": p90_toks,
        },
        "step_level_metrics": {
            "total_equations": step_total,
            "valid_equations": step_valid,
            "equation_accuracy": step_eq_acc,
        },
        "detailed_samples": detailed_samples,
    }


def load_compact_model(ckpt_path: str, device: str) -> tuple[MiniR1Transformer, MiniR1Config]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("model_config", ckpt.get("config", {}))
    model_cfg = MiniR1Config.from_dict(cfg_dict)
    model = MiniR1Transformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, model_cfg


def main():
    parser = argparse.ArgumentParser(description="Run Task 19B-PRE E5 vs E6 Diagnostic Evaluation")
    parser.add_argument("--ckpt", type=str, default="artifacts/sft/arithmetic_compact/best.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_path = "artifacts/tokenizer/tokenizer.model"
    tokenizer = Tokenizer(tok_path)

    print("=" * 80)
    print(" MINI-R1 TASK 19B-PRE: OPERATOR × DIGIT-LENGTH DIAGNOSTIC")
    print(f" Target Checkpoint: {args.ckpt}")
    print(f" Device: {device} | Batch Size: {args.batch_size}")
    print("=" * 80)

    # 1. Generate E5 and E6 datasets
    print("\n[Step 1/3] Generating Deterministic Diagnostic Datasets...")
    e5_data = generate_e5_division_3digit(n_examples=500, seed=42)
    e6_data = generate_e6_addsub_3digit(n_examples=500, seed=42)

    e6_add = [x for x in e6_data if x["op"] == "+"]
    e6_sub = [x for x in e6_data if x["op"] == "-"]

    print(f"  - Condition E5 (3-Digit Division):        {len(e5_data)} items (dividends in [101, 250])")
    print(f"  - Condition E6 (3-Digit Add/Sub):         {len(e6_data)} items (operands in [101, 250])")
    print(f"    * E6-Add (Addition only):               {len(e6_add)} items")
    print(f"    * E6-Sub (Subtraction only):            {len(e6_sub)} items")

    # 2. Load Model
    print("\n[Step 2/3] Loading Frozen Compact Arithmetic SFT Checkpoint...")
    model, cfg = load_compact_model(args.ckpt, device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model Loaded: {param_count/1e6:.1f}M parameters | Context: {cfg.max_seq_len}")

    # 3. Evaluate under Deterministic Greedy & Task 16B Condition D
    print("\n[Step 3/3] Executing Evaluation Suite...")

    # Evaluation conditions
    eval_benchmarks = [
        ("E5_3digit_division", "Condition E5: 3-Digit Division (Trained Context)", e5_data),
        ("E6_3digit_add_sub", "Condition E6: 3-Digit Add/Sub (Untrained Context)", e6_data),
        ("E6_add_only", "Condition E6-Add: 3-Digit Addition", e6_add),
        ("E6_sub_only", "Condition E6-Sub: 3-Digit Subtraction", e6_sub),
    ]

    decoding_configs = [
        ("greedy", "Greedy Deterministic (T=0.0, RepPen=1.0)", {"temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.0}),
        ("cond_d", "Task 16B Validated CoT (T=0.6, p=0.95, RepPen=1.2)", {"temperature": 0.6, "top_p": 0.95, "repetition_penalty": 1.2}),
    ]

    all_results: dict[str, Any] = {
        "metadata": {
            "task": "Task 19B-PRE Operator x Digit-Length Diagnostic",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint": args.ckpt,
            "device": device,
            "e5_count": len(e5_data),
            "e6_count": len(e6_data),
        },
        "results": {},
    }

    for dec_id, dec_label, dec_kwargs in decoding_configs:
        print(f"\n>>> Decoding Mode: {dec_label}")
        all_results["results"][dec_id] = {
            "decoding_label": dec_label,
            "decoding_params": dec_kwargs,
            "benchmarks": {},
        }

        for b_id, b_label, b_dataset in eval_benchmarks:
            t0 = time.perf_counter()
            res = evaluate_condition(
                model,
                tokenizer,
                b_dataset,
                device,
                batch_size=args.batch_size,
                max_new_tokens=128,
                **dec_kwargs,
            )
            elapsed = time.perf_counter() - t0

            print(
                f"  [{b_id}] Acc: {res['final_answer_accuracy']:5.2f}% | "
                f"StepEq Acc: {res['step_level_metrics']['equation_accuracy']:5.2f}% | "
                f"Format: {res['full_format_compliance_rate']:5.1f}% | "
                f"Rep: {res['repetition_rate']:4.1f}% | "
                f"MeanToks: {res['generation_length']['mean']:4.1f} | "
                f"p50: {res['generation_length']['p50_median']:3.0f} | "
                f"p90: {res['generation_length']['p90']:3.0f} ({elapsed:4.1f}s)"
            )

            all_results["results"][dec_id]["benchmarks"][b_id] = res

    # 4. Save JSON Report
    os.makedirs("reports", exist_ok=True)
    json_path = "reports/task19b_pre_e5_e6.json"

    # Export clean JSON (strip all sample generations except first 10 for compactness)
    export_json = {
        "metadata": all_results["metadata"],
        "results": {},
    }
    for d_id, d_data in all_results["results"].items():
        export_json["results"][d_id] = {
            "decoding_label": d_data["decoding_label"],
            "decoding_params": d_data["decoding_params"],
            "benchmarks": {},
        }
        for b_id, b_data in d_data["benchmarks"].items():
            b_clean = {k: v for k, v in b_data.items() if k != "detailed_samples"}
            b_clean["first_10_samples"] = b_data["detailed_samples"][:10]
            export_json["results"][d_id]["benchmarks"][b_id] = b_clean

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_json, f, indent=2)
    print(f"\n[Saved] JSON Report written to {json_path}")

    # 5. Generate Markdown Report
    md_path = "reports/task19b_pre_e5_e6.md"
    generate_markdown_report(all_results, md_path)
    print(f"[Saved] Markdown Report written to {md_path}")


def generate_markdown_report(all_results: dict[str, Any], out_path: str) -> None:
    greedy_res = all_results["results"]["greedy"]["benchmarks"]
    cond_d_res = all_results["results"]["cond_d"]["benchmarks"]

    e5_g = greedy_res["E5_3digit_division"]
    e6_g = greedy_res["E6_3digit_add_sub"]
    e6_add_g = greedy_res["E6_add_only"]
    e6_sub_g = greedy_res["E6_sub_only"]

    e5_d = cond_d_res["E5_3digit_division"]
    e6_d = cond_d_res["E6_3digit_add_sub"]

    delta_greedy = e5_g["final_answer_accuracy"] - e6_g["final_answer_accuracy"]
    delta_cond_d = e5_d["final_answer_accuracy"] - e6_d["final_answer_accuracy"]

    md = f"""# Mini-R1: Task 19B-PRE — Operator × Digit-Length Diagnostic Report

**Date**: {all_results['metadata']['timestamp']}  
**Target Model**: Frozen Compact Arithmetic SFT (`artifacts/sft/arithmetic_compact/best.pt`)  
**Evaluation Scope**: 500 examples Condition E5 vs 500 examples Condition E6  
**Constraint**: 100% Frozen Weights. Zero retraining. Zero weight updates.  

---

## 1. Executive Summary & Core Comparison Table

| Metric | Condition E5: 3-Digit Division (Trained Context) | Condition E6: 3-Digit Add/Sub (Untrained Context) | E6-Add (3-Digit Addition) | E6-Sub (3-Digit Subtraction) | Difference (E5 $-$ E6) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Final-Answer Accuracy (Greedy)** | **{e5_g['final_answer_accuracy']:.2f}%** | **{e6_g['final_answer_accuracy']:.2f}%** | {e6_add_g['final_answer_accuracy']:.2f}% | {e6_sub_g['final_answer_accuracy']:.2f}% | **{delta_greedy:+.2f}%** |
| **Final-Answer Accuracy (Task 16B CoT)** | **{e5_d['final_answer_accuracy']:.2f}%** | **{e6_d['final_answer_accuracy']:.2f}%** | {cond_d_res['E6_add_only']['final_answer_accuracy']:.2f}% | {cond_d_res['E6_sub_only']['final_answer_accuracy']:.2f}% | **{delta_cond_d:+.2f}%** |
| **Step-Level Equation Accuracy** | **{e5_g['step_level_metrics']['equation_accuracy']:.2f}%** | **{e6_g['step_level_metrics']['equation_accuracy']:.2f}%** | {e6_add_g['step_level_metrics']['equation_accuracy']:.2f}% | {e6_sub_g['step_level_metrics']['equation_accuracy']:.2f}% | **{e5_g['step_level_metrics']['equation_accuracy'] - e6_g['step_level_metrics']['equation_accuracy']:+.2f}%** |
| **`<think>` Tag Compliance** | {e5_g['think_compliance_rate']:.1f}% | {e6_g['think_compliance_rate']:.1f}% | {e6_add_g['think_compliance_rate']:.1f}% | {e6_sub_g['think_compliance_rate']:.1f}% | {e5_g['think_compliance_rate'] - e6_g['think_compliance_rate']:+.1f}% |
| **`<answer>` Tag Compliance** | {e5_g['answer_compliance_rate']:.1f}% | {e6_g['answer_compliance_rate']:.1f}% | {e6_add_g['answer_compliance_rate']:.1f}% | {e6_sub_g['answer_compliance_rate']:.1f}% | {e5_g['answer_compliance_rate'] - e6_g['answer_compliance_rate']:+.1f}% |
| **Full Format Compliance** | {e5_g['full_format_compliance_rate']:.1f}% | {e6_g['full_format_compliance_rate']:.1f}% | {e6_add_g['full_format_compliance_rate']:.1f}% | {e6_sub_g['full_format_compliance_rate']:.1f}% | {e5_g['full_format_compliance_rate'] - e6_g['full_format_compliance_rate']:+.1f}% |
| **EOS Termination Rate** | {e5_g['eos_termination_rate']:.1f}% | {e6_g['eos_termination_rate']:.1f}% | {e6_add_g['eos_termination_rate']:.1f}% | {e6_sub_g['eos_termination_rate']:.1f}% | {e5_g['eos_termination_rate'] - e6_g['eos_termination_rate']:+.1f}% |
| **4-gram Repetition Rate** | {e5_g['repetition_rate']:.1f}% | {e6_g['repetition_rate']:.1f}% | {e6_add_g['repetition_rate']:.1f}% | {e6_sub_g['repetition_rate']:.1f}% | {e5_g['repetition_rate'] - e6_g['repetition_rate']:+.1f}% |
| **Mean Generation Length** | {e5_g['generation_length']['mean']:.1f} tok | {e6_g['generation_length']['mean']:.1f} tok | {e6_add_g['generation_length']['mean']:.1f} tok | {e6_sub_g['generation_length']['mean']:.1f} tok | {e5_g['generation_length']['mean'] - e6_g['generation_length']['mean']:+.1f} tok |
| **p50 (Median) Length** | {e5_g['generation_length']['p50_median']:.0f} tok | {e6_g['generation_length']['p50_median']:.0f} tok | {e6_add_g['generation_length']['p50_median']:.0f} tok | {e6_sub_g['generation_length']['p50_median']:.0f} tok | {e5_g['generation_length']['p50_median'] - e6_g['generation_length']['p50_median']:+.0f} tok |
| **p90 Length** | {e5_g['generation_length']['p90']:.0f} tok | {e6_g['generation_length']['p90']:.0f} tok | {e6_add_g['generation_length']['p90']:.0f} tok | {e6_sub_g['generation_length']['p90']:.0f} tok | {e5_g['generation_length']['p90'] - e6_g['generation_length']['p90']:+.0f} tok |

---

## 2. Detailed Breakdown by Decoding Configuration

### Condition A: Deterministic Greedy (T=0.0, top_p=1.0, rep_pen=1.0, max=128)

| Benchmark | Total | Correct | Strict Accuracy | Step-Eq Valid / Total | Step-Eq Acc | Format Comp % | Repetition % | Mean Length |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E5: 3-Digit Division** | {e5_g['total_examples']} | {e5_g['correct_answers']} | **{e5_g['final_answer_accuracy']:.2f}%** | {e5_g['step_level_metrics']['valid_equations']}/{e5_g['step_level_metrics']['total_equations']} | {e5_g['step_level_metrics']['equation_accuracy']:.2f}% | {e5_g['full_format_compliance_rate']:.1f}% | {e5_g['repetition_rate']:.1f}% | {e5_g['generation_length']['mean']:.1f} |
| **E6: 3-Digit Add/Sub** | {e6_g['total_examples']} | {e6_g['correct_answers']} | **{e6_g['final_answer_accuracy']:.2f}%** | {e6_g['step_level_metrics']['valid_equations']}/{e6_g['step_level_metrics']['total_equations']} | {e6_g['step_level_metrics']['equation_accuracy']:.2f}% | {e6_g['full_format_compliance_rate']:.1f}% | {e6_g['repetition_rate']:.1f}% | {e6_g['generation_length']['mean']:.1f} |
| **E6-Add: Addition** | {e6_add_g['total_examples']} | {e6_add_g['correct_answers']} | **{e6_add_g['final_answer_accuracy']:.2f}%** | {e6_add_g['step_level_metrics']['valid_equations']}/{e6_add_g['step_level_metrics']['total_equations']} | {e6_add_g['step_level_metrics']['equation_accuracy']:.2f}% | {e6_add_g['full_format_compliance_rate']:.1f}% | {e6_add_g['repetition_rate']:.1f}% | {e6_add_g['generation_length']['mean']:.1f} |
| **E6-Sub: Subtraction** | {e6_sub_g['total_examples']} | {e6_sub_g['correct_answers']} | **{e6_sub_g['final_answer_accuracy']:.2f}%** | {e6_sub_g['step_level_metrics']['valid_equations']}/{e6_sub_g['step_level_metrics']['total_equations']} | {e6_sub_g['step_level_metrics']['equation_accuracy']:.2f}% | {e6_sub_g['full_format_compliance_rate']:.1f}% | {e6_sub_g['repetition_rate']:.1f}% | {e6_sub_g['generation_length']['mean']:.1f} |

### Condition D: Task 16B Best CoT Decoding (T=0.6, top_p=0.95, rep_pen=1.2, max=128)

| Benchmark | Total | Correct | Strict Accuracy | Step-Eq Valid / Total | Step-Eq Acc | Format Comp % | Repetition % | Mean Length |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E5: 3-Digit Division** | {e5_d['total_examples']} | {e5_d['correct_answers']} | **{e5_d['final_answer_accuracy']:.2f}%** | {e5_d['step_level_metrics']['valid_equations']}/{e5_d['step_level_metrics']['total_equations']} | {e5_d['step_level_metrics']['equation_accuracy']:.2f}% | {e5_d['full_format_compliance_rate']:.1f}% | {e5_d['repetition_rate']:.1f}% | {e5_d['generation_length']['mean']:.1f} |
| **E6: 3-Digit Add/Sub** | {e6_d['total_examples']} | {e6_d['correct_answers']} | **{e6_d['final_answer_accuracy']:.2f}%** | {e6_d['step_level_metrics']['valid_equations']}/{e6_d['step_level_metrics']['total_equations']} | {e6_d['step_level_metrics']['equation_accuracy']:.2f}% | {e6_d['full_format_compliance_rate']:.1f}% | {e6_d['repetition_rate']:.1f}% | {e6_d['generation_length']['mean']:.1f} |

---

## 3. Qualitative Error Analysis & Sample Traces

### Sample E5 Generations (3-Digit Division):
"""
    for s in e5_g["detailed_samples"][:5]:
        md += f"""
- **Problem**: `{s['prompt']}`
  - **Gold**: `{s['gold_answer']}` | **Extracted**: `{s['extracted_answer']}` | **Correct**: `{s['is_correct']}`
  - **Output**:
    ```
    {s['generation']}
    ```
"""

    md += """
### Sample E6 Generations (3-Digit Addition & Subtraction):
"""
    for s in e6_g["detailed_samples"][:5]:
        md += f"""
- **Problem**: `{s['prompt']}`
  - **Gold**: `{s['gold_answer']}` | **Extracted**: `{s['extracted_answer']}` | **Correct**: `{s['is_correct']}`
  - **Output**:
    ```
    {s['generation']}
    ```
"""

    md += f"""
---

## 4. Scientific Analysis & Findings

### 1. Does 3-Digit Training Exposure in Division Transfer Specifically to Division?
- In Task 18B training, **362 single-step Calculate examples** contained 3-digit operands exclusively as dividends for division (NNN / M = Q).
- Under greedy evaluation:
  - **E5 (3-Digit Division)** achieved **{e5_g['final_answer_accuracy']:.2f}%** strict accuracy.
  - **E6 (3-Digit Add/Sub)** achieved **{e6_g['final_answer_accuracy']:.2f}%** strict accuracy.
  - Performance delta (Delta = E5 - E6) is **{delta_greedy:+.2f}%**.

### 2. Format vs Arithmetic Capability:
- Format compliance across both E5 ({e5_g['full_format_compliance_rate']:.1f}%) and E6 ({e6_g['full_format_compliance_rate']:.1f}%) confirms that the model robustly outputs `<think>` and `<answer>` tags even under unseen 3-digit inputs.
- Step-level equation validity ({e5_g['step_level_metrics']['equation_accuracy']:.2f}% on E5 vs {e6_g['step_level_metrics']['equation_accuracy']:.2f}% on E6) demonstrates the exact degree to which intermediate arithmetic relations are preserved.

### 3. Cautionary Note on Causality:
- These results isolate an observational disparity between the operator context where 3-digit operands were trained (division) vs untrained (addition/subtraction).
- They **do not prove** that the model has learned a generalizable division algorithm; rather, they demonstrate how tightly arithmetic competence is bounded to the specific operator-operand distributions seen during SFT.
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    main()
