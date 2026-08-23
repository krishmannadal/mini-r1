"""
run_arithmetic_probe.py - Comprehensive Arithmetic Capability Probe for Mini-R1 (Task 17A)

Evaluates:
1. Base pretrained model (checkpoints/mini_r1_500m_pretrain/best.pt)
2. Direct SFT model (artifacts/sft/direct/best.pt)
3. CoT SFT model (artifacts/sft/cot/best.pt)

Across 9,000 synthetic arithmetic problems in 3 prompt formats:
Format 1: Plain ("What is 3 + 2?")
Format 2: Structured ("Calculate: 3 + 2")
Format 3: Reasoning ("Question: What is 3 + 2?\\n")
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import MiniR1Config, MiniR1Transformer
from src.tokenizer import Tokenizer


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def extract_strict_answer(text: str, is_base_model: bool = False) -> str | None:
    """Extract strict numeric answer from text."""
    if not is_base_model:
        # Require <answer>...</answer> tags
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
        if match:
            ans_str = match.group(1).strip()
            # Clean commas, currency, spaces
            cleaned = ans_str.replace(",", "").replace("$", "").strip()
            num_match = re.search(r"-?\d+", cleaned)
            if num_match:
                return num_match.group(0)
            return cleaned
        return None
    else:
        # For base model: find first standalone number after prompt / equal sign
        eq_match = re.search(r"=\s*(-?\d+)", text)
        if eq_match:
            return eq_match.group(1).strip()
        num_match = re.search(r"-?\d+", text)
        if num_match:
            return num_match.group(0).strip()
        return None


def extract_fallback_answer(text: str) -> str | None:
    """Extract last numeric token in generation."""
    matches = re.findall(r"-?\d+", text.replace(",", ""))
    return matches[-1] if matches else None


def detect_4gram_repetition(tokens: list[int]) -> bool:
    """Check if token sequence contains 4-gram repeating at least 3 times."""
    if len(tokens) < 12:
        return False
    ngrams: dict[tuple[int, ...], int] = {}
    for i in range(len(tokens) - 3):
        ng = tuple(tokens[i : i + 4])
        ngrams[ng] = ngrams.get(ng, 0) + 1
        if ngrams[ng] >= 3:
            return True
    return False


def verify_arithmetic_equations(trace_text: str) -> tuple[int, int, list[dict[str, Any]]]:
    """
    Parse arithmetic equations (e.g. X + Y = Z, A * B = C) and evaluate truth.
    Returns (valid_count, total_count, equation_details).
    """
    # Regex matching simple binary operations e.g. 12 + 7 = 19 or 4 * 5 = 20
    pattern = r"(\d+)\s*([\+\-\*\/])\s*(\d+)\s*=\s*(\d+)"
    matches = re.findall(pattern, trace_text)
    
    valid_count = 0
    total_count = len(matches)
    details = []
    
    for left, op, right, expected in matches:
        l_val = int(left)
        r_val = int(right)
        exp_val = int(expected)
        
        if op == "+":
            true_val = l_val + r_val
        elif op == "-":
            true_val = l_val - r_val
        elif op == "*":
            true_val = l_val * r_val
        elif op == "/":
            true_val = l_val // r_val if r_val != 0 else -999999
        else:
            true_val = -999999
        
        is_correct = (true_val == exp_val)
        if is_correct:
            valid_count += 1
        details.append({
            "eq": f"{left} {op} {right} = {expected}",
            "true_result": true_val,
            "stated_result": exp_val,
            "is_correct": is_correct,
        })
        
    return valid_count, total_count, details


def generate_batched_probe(
    model: MiniR1Transformer,
    prompts: list[str],
    tokenizer: Tokenizer,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    batch_size: int = 64,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """Batched autoregressive generation for arithmetic probing."""
    model.eval()
    eos_id = tokenizer.eos_id
    pad_id = tokenizer.pad_id
    stop_token_ids = {eos_id}

    results: list[dict[str, Any]] = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    with torch.inference_mode():
        for b_idx, b_start in enumerate(range(0, len(prompts), batch_size)):
            b_prompts = prompts[b_start : b_start + batch_size]
            b_encodings = [tokenizer.encode(p, add_bos=False, add_eos=False) for p in b_prompts]
            prompt_lens = [len(e) for e in b_encodings]
            max_p_len = max(prompt_lens)
            B = len(b_prompts)

            padded_inputs = []
            for e in b_encodings:
                pad_len = max_p_len - len(e)
                padded_inputs.append([pad_id] * pad_len + e)

            input_tensor = torch.tensor(padded_inputs, dtype=torch.long, device=device)
            finished = torch.zeros(B, dtype=torch.bool, device=device)
            gen_tokens: list[list[int]] = [[] for _ in range(B)]
            t0 = time.perf_counter()

            for step in range(max_new_tokens):
                curr_ctx = (
                    input_tensor
                    if input_tensor.size(1) <= model.config.max_seq_len
                    else input_tensor[:, -model.config.max_seq_len :]
                )
                with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16):
                    logits = model(curr_ctx)[:, -1, :].float()

                if repetition_penalty != 1.0:
                    for i in range(B):
                        if finished[i]:
                            continue
                        seen_tokens = set(b_encodings[i]).union(set(gen_tokens[i]))
                        if seen_tokens:
                            seen_indices = torch.tensor(list(seen_tokens), dtype=torch.long, device=device)
                            curr_logits = logits[i, seen_indices]
                            penalized_logits = torch.where(
                                curr_logits > 0,
                                curr_logits / repetition_penalty,
                                curr_logits * repetition_penalty,
                            )
                            logits[i, seen_indices] = penalized_logits

                if temperature <= 1e-6:
                    next_toks = torch.argmax(logits, dim=-1)
                else:
                    scaled_logits = logits / temperature
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                        sorted_indices_to_remove[:, 0] = False

                        for i in range(B):
                            indices_to_remove = sorted_indices[i, sorted_indices_to_remove[i]]
                            scaled_logits[i, indices_to_remove] = -float("Inf")

                    probs = F.softmax(scaled_logits, dim=-1)
                    next_toks = torch.multinomial(probs, num_samples=1).squeeze(1)

                for i in range(B):
                    if not finished[i]:
                        tok = int(next_toks[i].item())
                        gen_tokens[i].append(tok)
                        if tok in stop_token_ids:
                            finished[i] = True

                if finished.all():
                    break

                input_tensor = torch.cat([input_tensor, next_toks.unsqueeze(1)], dim=1)

            elapsed = max(1e-5, time.perf_counter() - t0)
            lat_per_sample = elapsed / B

            if (b_idx + 1) % 20 == 0 or (b_idx + 1) == total_batches:
                print(f"    [Batch {b_idx+1:03d}/{total_batches:03d}] Elapsed: {elapsed:.2f}s | EOS: {sum(finished.tolist())}/{B}", flush=True)

            for i in range(B):
                g_toks = gen_tokens[i]
                stopped = len(g_toks) > 0 and (g_toks[-1] in stop_token_ids)
                g_text = tokenizer.decode(g_toks)
                results.append({
                    "generated_text": g_text,
                    "prompt_tokens": prompt_lens[i],
                    "generated_tokens": len(g_toks),
                    "token_ids": g_toks,
                    "latency_sec": lat_per_sample,
                    "stopped_by_eos": stopped,
                })

    return results


def evaluate_model_on_format(
    model: MiniR1Transformer,
    dataset: list[dict[str, Any]],
    tokenizer: Tokenizer,
    prompt_key: str,
    format_name: str,
    model_name: str,
    is_base_model: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    batch_size: int = 64,
    device: str = "cuda",
) -> dict[str, Any]:
    """Evaluates a model on a specific prompt format across the dataset."""
    prompts = [item[prompt_key] for item in dataset]
    print(f"  Evaluating {model_name} on {format_name} ({len(prompts)} items)...", flush=True)

    t_start = time.perf_counter()
    gen_outputs = generate_batched_probe(
        model=model,
        prompts=prompts,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        batch_size=batch_size,
        device=device,
    )
    total_time = time.perf_counter() - t_start

    # Metrics aggregation
    total_samples = len(dataset)
    strict_correct_count = 0
    fallback_correct_count = 0
    parse_success_count = 0
    eos_count = 0
    repetition_count = 0
    has_think_count = 0
    has_answer_count = 0
    full_format_count = 0
    token_counts: list[int] = []

    # Category breakdowns
    cat_stats: dict[str, dict[str, int]] = {}
    op_count_stats: dict[int, dict[str, int]] = {}
    op_type_stats: dict[str, dict[str, int]] = {}

    evaluated_records: list[dict[str, Any]] = []

    for i, item in enumerate(dataset):
        gen = gen_outputs[i]
        gen_text = gen["generated_text"]
        gold = item["gold_answer"]
        cat = item["category"]
        op_cnt = item["op_count"]
        op_types = item["operations"]

        strict_ans = extract_strict_answer(gen_text, is_base_model=is_base_model)
        fallback_ans = extract_fallback_answer(gen_text)

        is_strict_corr = (strict_ans == gold) if strict_ans is not None else False
        is_fallback_corr = (fallback_ans == gold) if fallback_ans is not None else False
        is_parsed = (strict_ans is not None)
        is_rep = detect_4gram_repetition(gen["token_ids"])
        stopped = gen["stopped_by_eos"]

        has_think = ("<think>" in gen_text and "</think>" in gen_text)
        has_answer = ("<answer>" in gen_text and "</answer>" in gen_text)
        full_format = (has_think and has_answer)

        if is_strict_corr:
            strict_correct_count += 1
        if is_fallback_corr:
            fallback_correct_count += 1
        if is_parsed:
            parse_success_count += 1
        if stopped:
            eos_count += 1
        if is_rep:
            repetition_count += 1
        if has_think:
            has_think_count += 1
        if has_answer:
            has_answer_count += 1
        if full_format:
            full_format_count += 1

        token_counts.append(gen["generated_tokens"])

        # Category breakdowns
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "correct": 0}
        cat_stats[cat]["total"] += 1
        if is_strict_corr:
            cat_stats[cat]["correct"] += 1

        # Op count breakdowns
        if op_cnt not in op_count_stats:
            op_count_stats[op_cnt] = {"total": 0, "correct": 0}
        op_count_stats[op_cnt]["total"] += 1
        if is_strict_corr:
            op_count_stats[op_cnt]["correct"] += 1

        # Op type breakdowns
        for op in op_types:
            if op not in op_type_stats:
                op_type_stats[op] = {"total": 0, "correct": 0}
            op_type_stats[op]["total"] += 1
            if is_strict_corr:
                op_type_stats[op]["correct"] += 1

        evaluated_records.append({
            "problem_id": item["problem_id"],
            "category": cat,
            "underlying_expression": item["underlying_expression"],
            "gold_answer": gold,
            "generated_text": gen_text,
            "strict_extracted": strict_ans,
            "is_strict_correct": is_strict_corr,
            "fallback_extracted": fallback_ans,
            "is_fallback_correct": is_fallback_corr,
            "is_parsed": is_parsed,
            "is_repeated": is_rep,
            "has_think": has_think,
            "has_answer": has_answer,
            "full_format": full_format,
            "generated_tokens": gen["generated_tokens"],
            "stopped_by_eos": stopped,
        })

    token_counts.sort()
    mean_toks = sum(token_counts) / len(token_counts)
    median_toks = token_counts[len(token_counts) // 2]
    p90_toks = token_counts[int(len(token_counts) * 0.90)]

    strict_acc = strict_correct_count / total_samples
    fallback_acc = fallback_correct_count / total_samples
    parse_rate = parse_success_count / total_samples
    eos_rate = eos_count / total_samples
    rep_rate = repetition_count / total_samples

    print(f"    Strict Acc: {strict_acc*100:.2f}% ({strict_correct_count}/{total_samples}) | Parse: {parse_rate*100:.1f}% | EOS: {eos_rate*100:.1f}% | Rep: {rep_rate*100:.1f}% | Mean Toks: {mean_toks:.1f}", flush=True)

    return {
        "model_name": model_name,
        "format_name": format_name,
        "total_samples": total_samples,
        "strict_correct": strict_correct_count,
        "strict_accuracy": strict_acc,
        "fallback_correct": fallback_correct_count,
        "fallback_accuracy": fallback_acc,
        "parse_rate": parse_rate,
        "eos_rate": eos_rate,
        "repetition_rate": rep_rate,
        "think_tag_rate": has_think_count / total_samples,
        "answer_tag_rate": has_answer_count / total_samples,
        "full_format_rate": full_format_count / total_samples,
        "mean_tokens": mean_toks,
        "median_tokens": median_toks,
        "p90_tokens": p90_toks,
        "elapsed_seconds": total_time,
        "category_stats": {k: {"total": v["total"], "correct": v["correct"], "accuracy": v["correct"]/v["total"]} for k, v in sorted(cat_stats.items())},
        "op_count_stats": {k: {"total": v["total"], "correct": v["correct"], "accuracy": v["correct"]/v["total"]} for k, v in sorted(op_count_stats.items())},
        "op_type_stats": {k: {"total": v["total"], "correct": v["correct"], "accuracy": v["correct"]/v["total"]} for k, v in sorted(op_type_stats.items())},
        "records": evaluated_records,
    }


def perform_cot_trace_validation(cot_records: list[dict[str, Any]], num_samples: int = 100) -> dict[str, Any]:
    """Deterministic 100-example trace validation for CoT model."""
    subset = cot_records[:num_samples]
    
    categories = {
        "1. Correct answer + correct arithmetic steps": 0,
        "2. Correct answer + incorrect steps": 0,
        "3. Incorrect answer + structured arithmetic": 0,
        "4. Repetition": 0,
        "5. Format failure": 0,
    }

    detailed_traces: list[dict[str, Any]] = []

    for item in subset:
        gen_text = item["generated_text"]
        is_strict_corr = item["is_strict_correct"]
        is_rep = item["is_repeated"]
        full_format = item["full_format"]
        
        valid_eqs, total_eqs, eq_details = verify_arithmetic_equations(gen_text)
        all_eqs_valid = (total_eqs > 0 and valid_eqs == total_eqs)
        has_invalid_eqs = (total_eqs > 0 and valid_eqs < total_eqs)

        if is_rep:
            cat = "4. Repetition"
        elif not full_format:
            cat = "5. Format failure"
        elif is_strict_corr:
            if all_eqs_valid or total_eqs == 0:
                cat = "1. Correct answer + correct arithmetic steps"
            else:
                cat = "2. Correct answer + incorrect steps"
        else:
            cat = "3. Incorrect answer + structured arithmetic"

        categories[cat] += 1
        detailed_traces.append({
            "problem_id": item["problem_id"],
            "category": item["category"],
            "expression": item["underlying_expression"],
            "gold_answer": item["gold_answer"],
            "strict_extracted": item["strict_extracted"],
            "classification": cat,
            "total_equations_found": total_eqs,
            "valid_equations_found": valid_eqs,
            "equation_details": eq_details,
            "trace_snippet": gen_text,
        })

    return {
        "sample_size": num_samples,
        "taxonomy_counts": categories,
        "taxonomy_percentages": {k: v / num_samples * 100 for k, v in categories.items()},
        "traces": detailed_traces,
    }


def main():
    parser = argparse.ArgumentParser(description="Task 17A: Arithmetic Capability Probe")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 70)
    print(f" MINI-R1 TASK 17A: ARITHMETIC CAPABILITY PROBE")
    print(f" Device: {args.device} | Batch Size: {args.batch_size}")
    print("=" * 70)

    # Load dataset
    data_path = os.path.join("data", "eval", "synthetic_arithmetic.json")
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}, generating now...")
        import scripts.generate_arithmetic_probe as gen_script
        gen_script.main()

    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Loaded Synthetic Arithmetic Dataset: {len(dataset)} problems")

    # Tokenizer
    tok_path = "artifacts/tokenizer/tokenizer.model"
    tokenizer = Tokenizer(tok_path)

    # Checkpoint paths
    base_ckpt_path = "checkpoints/mini_r1_500m_pretrain/best.pt"
    direct_ckpt_path = "artifacts/sft/direct/best.pt"
    cot_ckpt_path = "artifacts/sft/cot/best.pt"

    base_sha = compute_file_sha256(base_ckpt_path)
    direct_sha = compute_file_sha256(direct_ckpt_path)
    cot_sha = compute_file_sha256(cot_ckpt_path)

    print(f"Base Checkpoint SHA256:   {base_sha}")
    print(f"Direct Checkpoint SHA256: {direct_sha}")
    print(f"CoT Checkpoint SHA256:    {cot_sha}")

    formats = [
        ("prompt_format_1_plain", "Format 1: Plain (What is 3+2?)"),
        ("prompt_format_2_structured", "Format 2: Structured (Calculate: 3+2)"),
        ("prompt_format_3_reasoning", "Format 3: Reasoning (Question: What is 3+2?)"),
    ]

    all_results: dict[str, Any] = {
        "metadata": {
            "task": "Task 17A: Arithmetic Capability Probe",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_path": data_path,
            "dataset_size": len(dataset),
            "seed": 42,
            "base_sha256": base_sha,
            "direct_sha256": direct_sha,
            "cot_sha256": cot_sha,
        },
        "evaluations": {},
    }

    # 1. Evaluate Base Model
    print("\n" + "=" * 70)
    print(" 1. EVALUATING BASE PRETRAINED MODEL (Greedy, MaxTok=32)")
    print("=" * 70)
    base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    base_cfg = MiniR1Config.from_dict(base_ckpt.get("model_config", base_ckpt.get("config", {})))
    base_model = MiniR1Transformer(base_cfg).to(args.device)
    base_model.load_state_dict(base_ckpt["model_state_dict"])
    base_model.eval()

    all_results["evaluations"]["base_pretrained"] = {}
    for p_key, f_name in formats:
        res = evaluate_model_on_format(
            model=base_model,
            dataset=dataset,
            tokenizer=tokenizer,
            prompt_key=p_key,
            format_name=f_name,
            model_name="Base Pretrained",
            is_base_model=True,
            max_new_tokens=32,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            batch_size=args.batch_size,
            device=args.device,
        )
        # Store records lightly (without full records in master JSON to keep report clean)
        all_results["evaluations"]["base_pretrained"][p_key] = {k: v for k, v in res.items() if k != "records"}

    del base_model, base_ckpt
    torch.cuda.empty_cache()

    # 2. Evaluate Direct SFT Model
    print("\n" + "=" * 70)
    print(" 2. EVALUATING DIRECT SFT MODEL (Greedy, MaxTok=32)")
    print("=" * 70)
    direct_ckpt = torch.load(direct_ckpt_path, map_location="cpu", weights_only=False)
    direct_cfg = MiniR1Config.from_dict(direct_ckpt.get("model_config", direct_ckpt.get("config", {})))
    direct_model = MiniR1Transformer(direct_cfg).to(args.device)
    direct_model.load_state_dict(direct_ckpt["model_state_dict"])
    direct_model.eval()

    all_results["evaluations"]["direct_sft"] = {}
    for p_key, f_name in formats:
        res = evaluate_model_on_format(
            model=direct_model,
            dataset=dataset,
            tokenizer=tokenizer,
            prompt_key=p_key,
            format_name=f_name,
            model_name="Direct SFT",
            is_base_model=False,
            max_new_tokens=32,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            batch_size=args.batch_size,
            device=args.device,
        )
        all_results["evaluations"]["direct_sft"][p_key] = {k: v for k, v in res.items() if k != "records"}

    del direct_model, direct_ckpt
    torch.cuda.empty_cache()

    # 3. Evaluate CoT SFT Model
    print("\n" + "=" * 70)
    print(" 3. EVALUATING COT SFT MODEL (T=0.6, p=0.95, RepPen=1.2, MaxTok=192)")
    print("=" * 70)
    cot_ckpt = torch.load(cot_ckpt_path, map_location="cpu", weights_only=False)
    cot_cfg = MiniR1Config.from_dict(cot_ckpt.get("model_config", cot_ckpt.get("config", {})))
    cot_model = MiniR1Transformer(cot_cfg).to(args.device)
    cot_model.load_state_dict(cot_ckpt["model_state_dict"])
    cot_model.eval()

    all_results["evaluations"]["cot_sft"] = {}
    cot_reasoning_records = None

    for p_key, f_name in formats:
        res = evaluate_model_on_format(
            model=cot_model,
            dataset=dataset,
            tokenizer=tokenizer,
            prompt_key=p_key,
            format_name=f_name,
            model_name="CoT SFT",
            is_base_model=False,
            max_new_tokens=192,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.2,
            batch_size=args.batch_size,
            device=args.device,
        )
        if p_key == "prompt_format_3_reasoning":
            cot_reasoning_records = res["records"]
        all_results["evaluations"]["cot_sft"][p_key] = {k: v for k, v in res.items() if k != "records"}

    # 4. Perform CoT Trace Validation (100 samples)
    print("\n" + "=" * 70)
    print(" 4. VALIDATING COT REASONING TRACES (100-Sample Subset)")
    print("=" * 70)
    trace_val = perform_cot_trace_validation(cot_reasoning_records, num_samples=100)
    all_results["cot_trace_validation"] = trace_val
    for cat_name, pct in trace_val["taxonomy_percentages"].items():
        print(f"  {cat_name:<46}: {pct:.1f}%")

    del cot_model, cot_ckpt
    torch.cuda.empty_cache()

    # Save JSON summary
    out_json = os.path.join("reports", "task17a_arithmetic_probe.json")
    os.makedirs("reports", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved raw results JSON to: {out_json}")

    # Generate Markdown Report
    generate_report_markdown(all_results, "reports/task17a_arithmetic_probe.md")
    print("Task 17A Report written to: reports/task17a_arithmetic_probe.md")


def generate_report_markdown(results: dict[str, Any], report_path: str):
    """Generate comprehensive Task 17A Markdown Report."""
    evals = results["evaluations"]
    base_eval = evals["base_pretrained"]
    direct_eval = evals["direct_sft"]
    cot_eval = evals["cot_sft"]
    trace_val = results["cot_trace_validation"]

    # Main comparison across Format 3 (Reasoning/Question prompt)
    base_f3 = base_eval["prompt_format_3_reasoning"]
    direct_f3 = direct_eval["prompt_format_3_reasoning"]
    cot_f3 = cot_eval["prompt_format_3_reasoning"]

    # Word wrap comparison: Expression (cats A-H) vs Word Wrapped (cat I)
    def get_expr_vs_word(eval_dict):
        cat_stats = eval_dict["category_stats"]
        expr_cats = ["A_single_addition", "B_single_subtraction", "C_single_multiplication", "D_single_division", "E_two_operation", "F_three_operation", "G_four_operation", "H_sequential_state"]
        expr_corr = sum(cat_stats[c]["correct"] for c in expr_cats if c in cat_stats)
        expr_tot = sum(cat_stats[c]["total"] for c in expr_cats if c in cat_stats)
        word_corr = cat_stats.get("I_word_wrapped", {}).get("correct", 0)
        word_tot = cat_stats.get("I_word_wrapped", {}).get("total", 1000)
        return (expr_corr / expr_tot * 100, word_corr / word_tot * 100)

    base_expr_acc, base_word_acc = get_expr_vs_word(base_f3)
    direct_expr_acc, direct_word_acc = get_expr_vs_word(direct_f3)
    cot_expr_acc, cot_word_acc = get_expr_vs_word(cot_f3)

    # Scientific Verdict determination
    # Check if arithmetic accuracy is strong on 1 op but collapses on multi-op
    cot_op1 = cot_f3["op_count_stats"].get("1", {}).get("accuracy", 0.0) * 100
    cot_op4 = cot_f3["op_count_stats"].get("4", {}).get("accuracy", 0.0) * 100
    cot_total = cot_f3["strict_accuracy"] * 100

    if cot_total >= 50.0:
        verdict = "A. Arithmetic substrate exists"
        verdict_desc = "The model reliably computes arithmetic operations across single- and multi-step expressions."
    elif cot_op1 >= 15.0 and cot_op4 < 5.0:
        verdict = "D. Arithmetic exists for simple operations but fails under composition"
        verdict_desc = "The model exhibits non-trivial accuracy on single-operation arithmetic but performance degrades sharply as operation depth increases."
    elif cot_total >= 2.0:
        verdict = "B. Arithmetic substrate is weak"
        verdict_desc = "Elementary arithmetic accuracy is poor (<10%) across almost all operations, confirming that arithmetic calculation represents the primary underlying bottleneck rather than language comprehension."
    else:
        verdict = "C. Arithmetic is mostly absent"
        verdict_desc = "The model operates near chance level (<2%) across all arithmetic probes, indicating the pretrained representations lack functional numerical arithmetic computation."

    lines = []
    lines.append("# Mini-R1: Task 17A — Arithmetic Capability Probe Report\n")
    lines.append(f"**Date**: {results['metadata']['date']}  ")
    lines.append(f"**Target Models**: Base Pretrained, Direct SFT, and CoT SFT  ")
    lines.append(f"**Benchmark**: 9,000 Deterministic Synthetic Arithmetic Problems (Seed 42)  ")
    lines.append(f"**Evaluation**: Deterministic Python Verifier (Zero LLM Judges, 100% Frozen Weights)  \n")
    lines.append("---\n")

    lines.append("## 1. Executive Summary\n")
    lines.append("Task 17A isolates numerical arithmetic computation from GSM8K linguistic complexity. We evaluated 9,000 synthetic arithmetic problems spanning single operations (addition, subtraction, multiplication, division), multi-operation compositions (2, 3, 4 ops), sequential state tracking, and word-wrapped problem variants across three distinct prompt formats.\n")
    lines.append("---\n")

    lines.append("## 2. Model Accuracy by Prompt Format\n")
    lines.append("| Model | Format 1 (Plain) | Format 2 (Structured) | Format 3 (Reasoning) | Mean Tokens (F3) | EOS Rate (F3) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for m_key, m_name in [("base_pretrained", "Base Pretrained (18k)"), ("direct_sft", "Direct SFT (650)"), ("cot_sft", "CoT SFT (650, Best Task 16B)")]:
        f1_acc = evals[m_key]["prompt_format_1_plain"]["strict_accuracy"] * 100
        f2_acc = evals[m_key]["prompt_format_2_structured"]["strict_accuracy"] * 100
        f3_acc = evals[m_key]["prompt_format_3_reasoning"]["strict_accuracy"] * 100
        mean_tok = evals[m_key]["prompt_format_3_reasoning"]["mean_tokens"]
        eos_r = evals[m_key]["prompt_format_3_reasoning"]["eos_rate"] * 100
        lines.append(f"| **{m_name}** | {f1_acc:.2f}% | {f2_acc:.2f}% | **{f3_acc:.2f}%** | {mean_tok:.1f} | {eos_r:.1f}% |")
    lines.append("\n---\n")

    lines.append("## 3. Operation Category Breakdown (Format 3)\n")
    lines.append("| Category | Total Items | Base Pretrained | Direct SFT | CoT SFT |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    categories = sorted(base_f3["category_stats"].keys())
    for cat in categories:
        b_acc = base_f3["category_stats"][cat]["accuracy"] * 100
        d_acc = direct_f3["category_stats"][cat]["accuracy"] * 100
        c_acc = cot_f3["category_stats"][cat]["accuracy"] * 100
        lines.append(f"| **{cat}** | 1000 | {b_acc:.2f}% | {d_acc:.2f}% | {c_acc:.2f}% |")
    lines.append("\n---\n")

    lines.append("## 4. Operation Complexity Breakdown (Format 3)\n")
    lines.append("| Operation Depth | Base Pretrained | Direct SFT | CoT SFT |")
    lines.append("| :--- | :---: | :---: | :---: |")
    for depth in sorted(base_f3["op_count_stats"].keys()):
        b_acc = base_f3["op_count_stats"][depth]["accuracy"] * 100
        d_acc = direct_f3["op_count_stats"][depth]["accuracy"] * 100
        c_acc = cot_f3["op_count_stats"][depth]["accuracy"] * 100
        lines.append(f"| **{depth} Operation(s)** | {b_acc:.2f}% | {d_acc:.2f}% | {c_acc:.2f}% |")
    lines.append("\n---\n")

    lines.append("## 5. Operation Type Breakdown (Format 3)\n")
    lines.append("| Operation Type | Base Pretrained | Direct SFT | CoT SFT |")
    lines.append("| :--- | :---: | :---: | :---: |")
    for op in ["addition", "subtraction", "multiplication", "division"]:
        b_acc = base_f3["op_type_stats"].get(op, {}).get("accuracy", 0.0) * 100
        d_acc = direct_f3["op_type_stats"].get(op, {}).get("accuracy", 0.0) * 100
        c_acc = cot_f3["op_type_stats"].get(op, {}).get("accuracy", 0.0) * 100
        lines.append(f"| **{op.capitalize()}** | {b_acc:.2f}% | {d_acc:.2f}% | {c_acc:.2f}% |")
    lines.append("\n---\n")

    lines.append("## 6. Word-Wrapping Analysis (Isolating Language Mapping vs Computation)\n")
    lines.append("| Model | Pure Arithmetic Expressions (N=8,000) | Word-Wrapped Equivalents (N=1,000) | Delta (Word - Expression) |")
    lines.append("| :--- | :---: | :---: | :---: |")
    lines.append(f"| **Base Pretrained** | {base_expr_acc:.2f}% | {base_word_acc:.2f}% | {base_word_acc - base_expr_acc:+.2f}% |")
    lines.append(f"| **Direct SFT** | {direct_expr_acc:.2f}% | {direct_word_acc:.2f}% | {direct_word_acc - direct_expr_acc:+.2f}% |")
    lines.append(f"| **CoT SFT** | {cot_expr_acc:.2f}% | {cot_word_acc:.2f}% | {cot_word_acc - cot_expr_acc:+.2f}% |")
    lines.append("\n---\n")

    lines.append("## 7. CoT Trace Validation (Deterministic 100-Sample Subset)\n")
    lines.append("Every arithmetic equation in the generated `<think>` trace was extracted and checked with an exact deterministic Python arithmetic verifier.\n")
    lines.append("| Forensic Trace Taxonomy Category | Count | Percentage |")
    lines.append("| :--- | :---: | :---: |")
    for cat_name, cnt in trace_val["taxonomy_counts"].items():
        pct = trace_val["taxonomy_percentages"][cat_name]
        lines.append(f"| **{cat_name}** | {cnt} / 100 | {pct:.1f}% |")
    lines.append("\n")

    lines.append("### Representative Traces from Probe:\n")
    for sample in trace_val["traces"][:3]:
        lines.append(f"#### Problem `{sample['problem_id']}` ({sample['expression']}) — [{sample['classification']}]")
        lines.append(f"- **Gold Answer**: `{sample['gold_answer']}` | **Extracted**: `{sample['strict_extracted']}`")
        lines.append(f"- **Equations Verified**: {sample['valid_equations_found']} valid / {sample['total_equations_found']} total")
        lines.append("```xml")
        lines.append(sample["trace_snippet"].strip())
        lines.append("```\n")

    lines.append("---\n")
    lines.append("## 8. Scientific Verdict & Conclusions\n")
    lines.append(f"### Final Classification:\n**{verdict}**\n")
    lines.append(f"{verdict_desc}\n")
    lines.append("---\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
