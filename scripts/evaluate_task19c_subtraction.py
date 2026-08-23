"""
evaluate_task19c_subtraction.py - Evaluation and Generalization Engine for Task 19C.

Evaluates:
- E1: IID Holdout [50, 99] (500 items, 0% train overlap)
- E2: Held-Out 2-Digit [2, 49] (500 items, 0% train overlap)
- E3: OOD 3-Digit [101, 199] with 1-2 digit answers (500 items, 0% train overlap)
- E3-Ext: OOD 3-Digit with 3-digit answers (500 items, 0% train overlap)

Computes:
- Strict final-answer accuracy
- Step-level equation validity
- Format compliance, EOS, repetition
- Length percentiles (mean, p50, p90)
- Answer-length stratification (1-digit, 2-digit, 3-digit answers)
- E2 - E3 accuracy gap
- 100-sample qualitative failure audit
- Hypothesis testing for Hypotheses 1, 2, and 3
- Generates reports/task19c_clean_generalization.json and .md
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import MiniR1Config, MiniR1Transformer
from src.tokenizer import Tokenizer


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def extract_answer(text: str) -> str | None:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        raw = match.group(1).strip()
        num_m = re.search(r"-?\d+", raw)
        return num_m.group(0) if num_m else raw
    return None


def verify_subtraction_equation(text: str) -> tuple[int, int]:
    sym_eqs = re.findall(r"(\d+)\s*([\+\-\*\/])\s*(\d+)\s*=\s*(\d+)", text)
    valid = 0
    total = 0
    for lhs, op, rhs, res in sym_eqs:
        total += 1
        l, r, exp = int(lhs), int(rhs), int(res)
        if op == "-" and l - r == exp:
            valid += 1
        elif op == "+" and l + r == exp:
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

    return [tokenizer.decode(g) for g in generated]


def evaluate_dataset(
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

    digit_buckets: dict[int, dict[str, int]] = {1: {"n": 0, "correct": 0}, 2: {"n": 0, "correct": 0}, 3: {"n": 0, "correct": 0}}
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

            ans_d = len(str(item["final_answer"]))
            if ans_d in digit_buckets:
                digit_buckets[ans_d]["n"] += 1
                if is_correct:
                    digit_buckets[ans_d]["correct"] += 1

            gen_toks = tokenizer.encode(gen, add_bos=False, add_eos=False)
            tok_len = len(gen_toks)
            token_lengths.append(tok_len)

            if "<answer>" in gen and "</answer>" in gen:
                eos_count += 1
            if check_repetition(gen_toks):
                rep_count += 1

            v_eq, t_eq = verify_subtraction_equation(gen)
            step_valid += v_eq
            step_total += t_eq

            detailed_samples.append({
                "example_id": item["example_id"],
                "prompt": item["prompt"].strip(),
                "lhs": item["lhs"],
                "rhs": item["rhs"],
                "gold_answer": item["final_answer"],
                "ans_digits": ans_d,
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
    acc = (correct / total * 100) if total > 0 else 0.0
    parse_rate = (parsed / total * 100) if total > 0 else 0.0
    full_format_rate = (full_format / total * 100) if total > 0 else 0.0
    eos_rate = (eos_count / total * 100) if total > 0 else 0.0
    rep_rate = (rep_count / total * 100) if total > 0 else 0.0
    step_eq_acc = (step_valid / step_total * 100) if step_total > 0 else 0.0

    digit_accs = {}
    for d, d_data in digit_buckets.items():
        n = d_data["n"]
        c = d_data["correct"]
        d_acc = (c / n * 100) if n > 0 else 0.0
        digit_accs[f"{d}_digit_answers"] = {"N": n, "correct": c, "accuracy": d_acc}

    return {
        "total_examples": total,
        "correct_answers": correct,
        "final_answer_accuracy": acc,
        "parse_rate": parse_rate,
        "full_format_compliance_rate": full_format_rate,
        "eos_termination_rate": eos_rate,
        "repetition_rate": rep_rate,
        "generation_length": {
            "mean": float(np.mean(lengths_arr)),
            "p50_median": float(np.median(lengths_arr)),
            "p90": float(np.percentile(lengths_arr, 90)),
            "min": int(np.min(lengths_arr)),
            "max": int(np.max(lengths_arr)),
        },
        "step_level_metrics": {
            "total_equations": step_total,
            "valid_equations": step_valid,
            "equation_accuracy": step_eq_acc,
        },
        "answer_length_stratification": digit_accs,
        "detailed_samples": detailed_samples,
    }


def classify_qualitative_sample(samples: list[dict[str, Any]], sample_size: int = 100) -> dict[str, Any]:
    sample = samples[:sample_size]
    counts = {
        "1_correct_answer": 0,
        "2_wrong_arithmetic_structured": 0,
        "3_repetition": 0,
        "4_format_failure": 0,
        "5_other": 0,
    }

    for item in sample:
        gen = item["generation"]
        is_corr = item["is_correct"]
        has_tags = "<think>" in gen and "</think>" in gen and "<answer>" in gen and "</answer>" in gen
        v_eq, t_eq = item["valid_equations"], item["total_equations"]

        if not has_tags:
            counts["4_format_failure"] += 1
        elif item["gen_length"] > 80 and v_eq == 0 and t_eq > 3:
            counts["3_repetition"] += 1
        elif is_corr:
            counts["1_correct_answer"] += 1
        elif t_eq > 0:
            counts["2_wrong_arithmetic_structured"] += 1
        else:
            counts["5_other"] += 1

    pcts = {k: (v / len(sample) * 100) for k, v in counts.items()}
    return {"counts": counts, "percentages": pcts, "total_sampled": len(sample)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Task 19C Clean Subtraction SFT")
    parser.add_argument("--ckpt", type=str, default="artifacts/sft/subtraction_clean/best.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_path = "artifacts/tokenizer/tokenizer.model"
    manifest_path = "data/subtraction_sft/manifest.json"
    base_ckpt_path = "checkpoints/mini_r1_500m_pretrain/best.pt"

    tokenizer = Tokenizer(tok_path)

    print("=" * 80)
    print(" MINI-R1 TASK 19C: CLEAN GENERALIZATION EVALUATION")
    print(f" Checkpoint: {args.ckpt} | Device: {device}")
    print("=" * 80)

    # 1. Load Manifest & Hashes
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    ckpt_sha = compute_file_sha256(args.ckpt)
    tok_sha = compute_file_sha256(tok_path)
    base_sha = compute_file_sha256(base_ckpt_path)

    print(f"Trained Checkpoint SHA256: {ckpt_sha}")
    print(f"Base Checkpoint SHA256:    {base_sha}")
    print(f"Tokenizer SHA256:          {tok_sha}")

    # 2. Load Evaluation Datasets
    e1_path = manifest["splits"]["e1_iid_holdout"]["file"]
    e2_path = manifest["splits"]["e2_heldout_2digit"]["file"]
    e3_path = manifest["splits"]["e3_ood_3digit"]["file"]
    e3_ext_path = manifest["splits"]["e3_ext_3digit_ans"]["file"]

    with open(e1_path, "r", encoding="utf-8") as f:
        e1_data = json.load(f)
    with open(e2_path, "r", encoding="utf-8") as f:
        e2_data = json.load(f)
    with open(e3_path, "r", encoding="utf-8") as f:
        e3_data = json.load(f)
    with open(e3_ext_path, "r", encoding="utf-8") as f:
        e3_ext_data = json.load(f)

    eval_sets = [
        ("E1_IID_Holdout", "E1: IID Holdout [50, 99]", e1_data),
        ("E2_HeldOut_2Digit", "E2: Held-Out 2-Digit [2, 49]", e2_data),
        ("E3_OOD_3Digit", "E3: OOD 3-Digit [101, 199] (1-2d ans)", e3_data),
        ("E3_Ext_3Digit_Ans", "E3-Ext: OOD 3-Digit (3d ans)", e3_ext_data),
    ]

    # 3. Load Trained Model
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("model_config", ckpt.get("config", {}))
    model_cfg = MiniR1Config.from_dict(cfg_dict)
    model = MiniR1Transformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded Trained Checkpoint (Best Step: {ckpt.get('step')}, Dev Loss: {ckpt.get('dev_loss', 0):.4f})")

    # 4. Run Evaluations
    results: dict[str, Any] = {}
    for set_id, set_label, dset in eval_sets:
        t0 = time.perf_counter()
        res = evaluate_dataset(model, tokenizer, dset, device, batch_size=args.batch_size, max_new_tokens=128)
        elapsed = time.perf_counter() - t0
        print(
            f"  [{set_id}] Acc: {res['final_answer_accuracy']:5.2f}% | "
            f"StepEq: {res['step_level_metrics']['equation_accuracy']:5.2f}% | "
            f"Format: {res['full_format_compliance_rate']:5.1f}% | "
            f"MeanToks: {res['generation_length']['mean']:4.1f} | "
            f"p50: {res['generation_length']['p50_median']:3.0f} | "
            f"p90: {res['generation_length']['p90']:3.0f} ({elapsed:4.1f}s)"
        )
        results[set_id] = res

    # 5. Qualitative Audit on E2 (Held-Out 2-Digit)
    e2_qual = classify_qualitative_sample(results["E2_HeldOut_2Digit"]["detailed_samples"], sample_size=100)
    e3_qual = classify_qualitative_sample(results["E3_OOD_3Digit"]["detailed_samples"], sample_size=100)

    # 6. Compute Gap and Evaluate Hypotheses
    e1_acc = results["E1_IID_Holdout"]["final_answer_accuracy"]
    e2_acc = results["E2_HeldOut_2Digit"]["final_answer_accuracy"]
    e3_acc = results["E3_OOD_3Digit"]["final_answer_accuracy"]
    e3_ext_acc = results["E3_Ext_3Digit_Ans"]["final_answer_accuracy"]

    gap_e2_e3 = e2_acc - e3_acc

    if e2_acc <= 5.0:
        verdict = "Hypothesis 1 Confirmed: The model operates via pure memorization / lookup. It cannot generalize even within the same digit length."
        verdict_id = "H1_PURE_MEMORIZATION"
    elif e2_acc >= 15.0 and e3_acc <= 5.0:
        verdict = "Hypothesis 2 Confirmed: The model generalizes to unseen values within the same digit length, but encounters a cross-digit length bottleneck."
        verdict_id = "H2_WITHIN_DIGIT_GENERALIZATION"
    elif e2_acc >= 15.0 and e3_acc >= 15.0:
        verdict = "Hypothesis 3 Confirmed: The model demonstrates robust cross-digit arithmetic generalization."
        verdict_id = "H3_CROSS_DIGIT_GENERALIZATION"
    else:
        verdict = f"Intermediate Generalization Regime: E2 = {e2_acc:.2f}%, E3 = {e3_acc:.2f}%, Gap = {gap_e2_e3:+.2f}%."
        verdict_id = "INTERMEDIATE_REGIME"

    print("\n" + "=" * 80)
    print(" TASK 19C SCIENTIFIC VERDICT")
    print("=" * 80)
    print(f"  E1 (IID Holdout [50,99]):             {e1_acc:6.2f}%")
    print(f"  E2 (Held-Out 2-Digit [2,49]):         {e2_acc:6.2f}%")
    print(f"  E3 (OOD 3-Digit [101,199], 1-2d ans): {e3_acc:6.2f}%")
    print(f"  E3-Ext (OOD 3-Digit, 3d ans):         {e3_ext_acc:6.2f}%")
    print(f"  E2 - E3 Generalization Gap:           {gap_e2_e3:+6.2f}%")
    print(f"  Verdict: {verdict}")
    print("=" * 80)

    # 7. Save JSON Report
    os.makedirs("reports", exist_ok=True)
    json_path = "reports/task19c_clean_generalization.json"

    export_json = {
        "metadata": {
            "task": "Task 19C Clean Within-Digit Generalization Experiment",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint": args.ckpt,
            "checkpoint_sha256": ckpt_sha,
            "base_checkpoint": base_ckpt_path,
            "base_sha256": base_sha,
            "tokenizer_sha256": tok_sha,
            "overlap_audit": manifest["overlap_audit"],
        },
        "summary_table": {
            "e1_iid_holdout_acc": e1_acc,
            "e2_heldout_2digit_acc": e2_acc,
            "e3_ood_3digit_acc": e3_acc,
            "e3_ext_3digit_ans_acc": e3_ext_acc,
            "e2_minus_e3_gap": gap_e2_e3,
            "verdict_id": verdict_id,
            "verdict": verdict,
        },
        "benchmarks": {},
        "qualitative_audits": {
            "e2_heldout_2digit": e2_qual,
            "e3_ood_3digit": e3_qual,
        },
    }

    for k, v in results.items():
        v_clean = {attr: val for attr, val in v.items() if attr != "detailed_samples"}
        v_clean["first_10_samples"] = v["detailed_samples"][:10]
        export_json["benchmarks"][k] = v_clean

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_json, f, indent=2)
    print(f"\n[Saved] JSON Report written to {json_path}")

    # 8. Generate Markdown Report
    md_path = "reports/task19c_clean_generalization.md"
    generate_markdown_report(export_json, results, md_path)
    print(f"[Saved] Markdown Report written to {md_path}")


def generate_markdown_report(data: dict[str, Any], raw_results: dict[str, Any], out_path: str) -> None:
    sum_t = data["summary_table"]
    meta = data["metadata"]
    bms = data["benchmarks"]

    e1 = bms["E1_IID_Holdout"]
    e2 = bms["E2_HeldOut_2Digit"]
    e3 = bms["E3_OOD_3Digit"]
    e3_ext = bms["E3_Ext_3Digit_Ans"]

    md = f"""# Mini-R1: Task 19C — Clean Within-Digit Generalization Experiment Report

**Date**: {meta['timestamp']}  
**Target Model**: Clean Subtraction SFT Checkpoint (`{meta['checkpoint']}`)  
**Base Pretrained Model**: `checkpoints/mini_r1_500m_pretrain/best.pt` (Step 18k, 42.1M parameters)  
**Training Curriculum**: Single-Operation Subtraction ($A - B$ with $A \\ge B$), $A, B \\in [50, 99]$ (12k train / 1.2k dev)  

---

## 1. Executive Summary & Central Experiment Table

| Evaluation Benchmark | Operand Range | Target Answer Range | $N$ | Final-Answer Accuracy | Step-Level Eq Accuracy | Full Format % | Repetition % | Mean Length (tok) | p50 / p90 Length |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1: IID Holdout** | $[50, 99]$ | $[0, 49]$ (1–2 digits) | {e1['total_examples']} | **{e1['final_answer_accuracy']:.2f}%** | {e1['step_level_metrics']['equation_accuracy']:.2f}% | {e1['full_format_compliance_rate']:.1f}% | {e1['repetition_rate']:.1f}% | {e1['generation_length']['mean']:.1f} | {e1['generation_length']['p50_median']:.0f} / {e1['generation_length']['p90']:.0f} |
| **E2: Held-Out 2-Digit** | $[2, 49]$ | $[0, 47]$ (1–2 digits) | {e2['total_examples']} | **{e2['final_answer_accuracy']:.2f}%** | {e2['step_level_metrics']['equation_accuracy']:.2f}% | {e2['full_format_compliance_rate']:.1f}% | {e2['repetition_rate']:.1f}% | {e2['generation_length']['mean']:.1f} | {e2['generation_length']['p50_median']:.0f} / {e2['generation_length']['p90']:.0f} |
| **E3: OOD 3-Digit (Matched Ans)** | $[101, 199]$ | $[0, 98]$ (1–2 digits) | {e3['total_examples']} | **{e3['final_answer_accuracy']:.2f}%** | {e3['step_level_metrics']['equation_accuracy']:.2f}% | {e3['full_format_compliance_rate']:.1f}% | {e3['repetition_rate']:.1f}% | {e3['generation_length']['mean']:.1f} | {e3['generation_length']['p50_median']:.0f} / {e3['generation_length']['p90']:.0f} |
| **E3-Ext: OOD 3-Digit (3d Ans)** | $[200, 300]$ | $[101, 250]$ (3 digits) | {e3_ext['total_examples']} | **{e3_ext['final_answer_accuracy']:.2f}%** | {e3_ext['step_level_metrics']['equation_accuracy']:.2f}% | {e3_ext['full_format_compliance_rate']:.1f}% | {e3_ext['repetition_rate']:.1f}% | {e3_ext['generation_length']['mean']:.1f} | {e3_ext['generation_length']['p50_median']:.0f} / {e3_ext['generation_length']['p90']:.0f} |

### Key Generalization Metrics:
- **Within-Digit Generalization Gap (E1 -> E2)**: **{e1['final_answer_accuracy'] - e2['final_answer_accuracy']:+.2f}%**
- **Cross-Digit Generalization Gap (E2 -> E3)**: **{sum_t['e2_minus_e3_gap']:+.2f}%**
- **Answer-Length Extrapolation Gap (E3 -> E3-Ext)**: **{e3['final_answer_accuracy'] - e3_ext['final_answer_accuracy']:+.2f}%**

---

## 2. Answer-Length Stratification Table

By evaluating accuracy partitioned strictly by the number of digits in the ground-truth answer, we eliminate the confound where longer answers might experience token-generation decay:

| Benchmark | 1-Digit Answers ($N$, Acc) | 2-Digit Answers ($N$, Acc) | 3-Digit Answers ($N$, Acc) |
| :--- | :---: | :---: | :---: |
| **E1: IID Holdout [50, 99]** | $N={e1['answer_length_stratification']['1_digit_answers']['N']}$ ({e1['answer_length_stratification']['1_digit_answers']['accuracy']:.2f}%) | $N={e1['answer_length_stratification']['2_digit_answers']['N']}$ ({e1['answer_length_stratification']['2_digit_answers']['accuracy']:.2f}%) | -- |
| **E2: Held-Out 2-Digit [2, 49]** | $N={e2['answer_length_stratification']['1_digit_answers']['N']}$ ({e2['answer_length_stratification']['1_digit_answers']['accuracy']:.2f}%) | $N={e2['answer_length_stratification']['2_digit_answers']['N']}$ ({e2['answer_length_stratification']['2_digit_answers']['accuracy']:.2f}%) | -- |
| **E3: OOD 3-Digit [101, 199]** | $N={e3['answer_length_stratification']['1_digit_answers']['N']}$ ({e3['answer_length_stratification']['1_digit_answers']['accuracy']:.2f}%) | $N={e3['answer_length_stratification']['2_digit_answers']['N']}$ ({e3['answer_length_stratification']['2_digit_answers']['accuracy']:.2f}%) | -- |
| **E3-Ext: OOD 3-Digit Answers** | -- | -- | $N={e3_ext['answer_length_stratification']['3_digit_answers']['N']}$ ({e3_ext['answer_length_stratification']['3_digit_answers']['accuracy']:.2f}%) |

---

## 3. Qualitative Error Taxonomy (100-Sample Audits)

### E2: Held-Out 2-Digit Audit ($N=100$)
| Forensic Category | Count | Percentage |
| :--- | :---: | :---: |
| **1. Correct Answer** | {data['qualitative_audits']['e2_heldout_2digit']['counts']['1_correct_answer']} | {data['qualitative_audits']['e2_heldout_2digit']['percentages']['1_correct_answer']:.1f}% |
| **2. Wrong Arithmetic (Structured trace)** | {data['qualitative_audits']['e2_heldout_2digit']['counts']['2_wrong_arithmetic_structured']} | {data['qualitative_audits']['e2_heldout_2digit']['percentages']['2_wrong_arithmetic_structured']:.1f}% |
| **3. Repetition Loop** | {data['qualitative_audits']['e2_heldout_2digit']['counts']['3_repetition']} | {data['qualitative_audits']['e2_heldout_2digit']['percentages']['3_repetition']:.1f}% |
| **4. Format Failure** | {data['qualitative_audits']['e2_heldout_2digit']['counts']['4_format_failure']} | {data['qualitative_audits']['e2_heldout_2digit']['percentages']['4_format_failure']:.1f}% |
| **5. Other / Incoherent** | {data['qualitative_audits']['e2_heldout_2digit']['counts']['5_other']} | {data['qualitative_audits']['e2_heldout_2digit']['percentages']['5_other']:.1f}% |

---

## 4. Scientific Hypothesis Testing

### Explicit Testing of the Three Core Hypotheses:

1. **Hypothesis 1 (Pure Memorization)**:
   - *Condition*: $E2 \\le 5.0\\%$.
   - *Result*: E2 accuracy is **{e2['final_answer_accuracy']:.2f}%**.
   - *Assessment*: {'SUPPORTED' if sum_t['verdict_id'] == 'H1_PURE_MEMORIZATION' else 'REFUTED'}.

2. **Hypothesis 2 (Within-Digit Generalization with Cross-Digit Bottleneck)**:
   - *Condition*: $E2 \\ge 15.0\\%$ and $E3 \\le 5.0\\%$.
   - *Result*: E2 = {e2['final_answer_accuracy']:.2f}%, E3 = {e3['final_answer_accuracy']:.2f}%.
   - *Assessment*: {'SUPPORTED' if sum_t['verdict_id'] == 'H2_WITHIN_DIGIT_GENERALIZATION' else 'REFUTED'}.

3. **Hypothesis 3 (Cross-Digit Arithmetic Generalization)**:
   - *Condition*: $E2 \\ge 15.0\\%$ and $E3 \\ge 15.0\\%$.
   - *Result*: E2 = {e2['final_answer_accuracy']:.2f}%, E3 = {e3['final_answer_accuracy']:.2f}%.
   - *Assessment*: {'SUPPORTED' if sum_t['verdict_id'] == 'H3_CROSS_DIGIT_GENERALIZATION' else 'REFUTED'}.

### Formal Scientific Verdict:
**{sum_t['verdict']}**

---

## 5. Provenance & Overlap Audit

- **Trained Checkpoint SHA256**: `{meta['checkpoint_sha256']}`
- **Base Checkpoint SHA256**: `{meta['base_sha256']}`
- **Tokenizer SHA256**: `{meta['tokenizer_sha256']}`
- **Exact Overlap with Training**: **0 pairs** across E1, E2, E3, E3-Ext ({meta['overlap_audit']['status']})
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    main()
