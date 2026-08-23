"""
evaluate_task22_replay.py - Task 22 evaluation harness.

Reuses the Task 21B harness (scripts/evaluate_task21b_transfer.py) for answer
extraction, repetition detection, benchmark loading and general-language PPL.
Generation keeps the Task 21B batching contract EXACTLY (BOS-prefixed prompt,
left padding, no attention mask passed to the model, max_new_tokens=256); the
only thing that changes is the token-selection rule, so decoding is a clean
isolated variable.

NEW PRIMARY METRIC -- STEP-LEVEL EQUATION VALIDITY
    (# generated equations that verify true) / (total generated equations)
computed over the <think> blocks of the GSM8K test generations, with a
deterministic Python verifier. It is reported for the Task 22 model AND
recomputed from scratch for Task 21B Model A and Model B, so the comparison is
apples-to-apples rather than lifted from the Task 21B report (which never
measured it).

Decoding conditions:
    PRIMARY   : T=0.6, top_p=0.95, repetition_penalty=1.2   (Task 16B best)
    SECONDARY : greedy (argmax)                             (Task 21B's setting)

Also reported: GSM8K strict accuracy, 2D/3D/4D subtraction retention,
held-out replay-regime arithmetic accuracy, and general-language perplexity.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
for _p in (PROJECT_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- reuse the Task 21B harness; nothing in it is modified ------------------
from evaluate_task21b_transfer import (  # noqa: E402
    check_repetition,
    compute_sha256,
    extract_strict_answer,
)
from src.evaluation import evaluate_general_language  # noqa: E402
from src.datasets import ReasoningDatasetManager  # noqa: E402
from src.model import MiniR1Config, MiniR1Transformer  # noqa: E402
from src.tokenizer import Tokenizer  # noqa: E402


DECODE_CONFIGS = {
    "sampled_t06_p095_rp12": {
        "label": "T=0.6, top_p=0.95, rep_pen=1.2 (Task 16B best)",
        "temperature": 0.6,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "role": "primary",
    },
    "greedy": {
        "label": "greedy argmax (Task 21B setting)",
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "role": "secondary",
    },
}


# --------------------------------------------------------------------------- #
# Step-level equation validity
# --------------------------------------------------------------------------- #
EQ_LINE_RE = re.compile(r"^[\s\d\.\+\-\*\/\(\)]*=[\s\-\d\.]+$")
SAFE_LHS_RE = re.compile(r"^[\d\.\s\+\-\*\/\(\)]+$")


def extract_think_block(gen: str) -> str:
    """The reasoning region of a generation, tolerant of missing open/close tags."""
    m = re.search(r"<think>(.*?)</think>", gen, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"<think>(.*)", gen, re.DOTALL)
    if m:
        body = m.group(1)
        return body.split("<answer>")[0]
    return gen.split("<answer>")[0]


def verify_equation(line: str) -> tuple[bool, bool]:
    """
    Returns (is_equation, verifies_true).

    An equation is a line of the form '<arithmetic expression> = <number>'.
    It verifies true when deterministic Python evaluation of the LHS equals the
    stated RHS (tolerance 1e-4, matching the compact GSM8K dataset builder).
    """
    s = line.strip()
    if not s or s.count("=") != 1:
        return False, False
    if not EQ_LINE_RE.match(s):
        return False, False
    lhs, rhs = s.split("=")
    lhs, rhs = lhs.strip(), rhs.strip()
    if not lhs or not rhs:
        return False, False
    if not SAFE_LHS_RE.match(lhs) or not re.search(r"\d", lhs):
        return False, False
    try:
        rv = float(rhs)
    except ValueError:
        return False, False
    # An LHS with no operator ('5 = 5') is a restatement, not a computation.
    if not re.search(r"[\+\-\*\/]", lhs.strip().lstrip("-")):
        return False, False
    try:
        lv = eval(lhs, {"__builtins__": None}, {})
    except Exception:
        return True, False  # well-formed equation shape, but not evaluable
    if isinstance(lv, complex) or lv is None:
        return True, False
    try:
        return True, abs(float(lv) - rv) < 1e-4
    except (TypeError, ValueError, OverflowError):
        return True, False


def equation_validity(generations: list[str], extracted: list) -> dict[str, Any]:
    """Step-level equation validity over a set of generations."""
    total_eq = 0
    valid_eq = 0
    gens_with_eq = 0
    per_gen_counts: list[int] = []
    final_step_consistent = 0
    final_step_checked = 0
    invalid_examples: list[str] = []

    for gen, ans in zip(generations, extracted):
        block = extract_think_block(gen)
        eqs: list[tuple[str, bool]] = []
        for raw in block.split("\n"):
            is_eq, ok = verify_equation(raw)
            if is_eq:
                eqs.append((raw.strip(), ok))
        per_gen_counts.append(len(eqs))
        if eqs:
            gens_with_eq += 1
        for text, ok in eqs:
            total_eq += 1
            if ok:
                valid_eq += 1
            elif len(invalid_examples) < 15:
                invalid_examples.append(text)
        # does the last emitted equation agree with the <answer> tag?
        if eqs and ans is not None:
            final_step_checked += 1
            try:
                rhs = float(eqs[-1][0].split("=")[1].strip())
                if abs(rhs - float(ans)) < 1e-4:
                    final_step_consistent += 1
            except (ValueError, IndexError):
                pass

    n = len(generations)
    return {
        "total_generated_equations": total_eq,
        "valid_generated_equations": valid_eq,
        "step_level_equation_validity": (valid_eq / total_eq * 100.0) if total_eq else 0.0,
        "generations_with_at_least_one_equation": gens_with_eq,
        "pct_generations_with_equation": (gens_with_eq / n * 100.0) if n else 0.0,
        "mean_equations_per_generation": float(np.mean(per_gen_counts)) if per_gen_counts else 0.0,
        # Absolute counterpart to the rate. The rate alone is denominator-sensitive:
        # a model that emits almost no equations can post a high validity rate on the
        # few it does emit. This measures verified-correct arithmetic actually produced.
        "valid_equations_per_generation": (valid_eq / n) if n else 0.0,
        "final_step_answer_consistency": (
            final_step_consistent / final_step_checked * 100.0 if final_step_checked else 0.0
        ),
        "final_step_checked": final_step_checked,
        "invalid_equation_examples": invalid_examples,
    }


# --------------------------------------------------------------------------- #
# Generation (Task 21B batching contract + configurable token selection)
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def batch_generate_decoded(
    model: MiniR1Transformer,
    tokenizer: Tokenizer,
    prompts: list[str],
    device: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    generator: torch.Generator | None = None,
    stop_ids: set[int] | None = None,
) -> list[str]:
    encoded = [tokenizer.encode(p, add_bos=True, add_eos=False) for p in prompts]
    max_p_len = max(len(x) for x in encoded)
    pad_id = tokenizer.pad_id
    eos_id = tokenizer.eos_id
    stop_ids = stop_ids if stop_ids is not None else {eos_id}

    padded_in = [[pad_id] * (max_p_len - len(s)) + s for s in encoded]
    input_tensor = torch.tensor(padded_in, dtype=torch.long, device=device)
    B = input_tensor.size(0)
    generated: list[list[int]] = [[] for _ in range(B)]
    finished = [False] * B
    max_ctx = model.config.max_seq_len

    for _ in range(max_new_tokens):
        if all(finished):
            break
        ctx = input_tensor if input_tensor.size(1) <= max_ctx else input_tensor[:, -max_ctx:]
        with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16):
            logits = model(ctx)
        next_logits = logits[:, -1, :].float()

        if repetition_penalty != 1.0:
            for b in range(B):
                if finished[b]:
                    continue
                seen = set(encoded[b]) | set(generated[b])
                if not seen:
                    continue
                idx = torch.tensor(sorted(seen), dtype=torch.long, device=device)
                cur = next_logits[b, idx]
                next_logits[b, idx] = torch.where(
                    cur > 0, cur / repetition_penalty, cur * repetition_penalty
                )

        if temperature <= 1e-6:
            next_tokens = torch.argmax(next_logits, dim=-1)
        else:
            scaled = next_logits / temperature
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
                cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                scaled = scaled.scatter(1, sorted_idx, sorted_logits.masked_fill(remove, -float("inf")))
            probs = F.softmax(scaled, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)

        for b in range(B):
            if not finished[b]:
                tok = int(next_tokens[b].item())
                if tok == eos_id:
                    finished[b] = True
                else:
                    generated[b].append(tok)
                    if tok in stop_ids:
                        # </answer> closes the completion. Everything the Task 21B
                        # harness measures downstream reads the FIRST <think>..</think>
                        # and the FIRST <answer>..</answer>, so stopping here yields
                        # byte-identical equation metrics, strict accuracy and format
                        # compliance while cutting generation cost dramatically.
                        finished[b] = True

        input_tensor = torch.cat([input_tensor, next_tokens.unsqueeze(1)], dim=1)

    return [tokenizer.decode(g) for g in generated]


def run_benchmark(
    model: MiniR1Transformer,
    tokenizer: Tokenizer,
    problems: list[dict[str, Any]],
    name: str,
    device: str,
    decode: dict[str, Any],
    batch_size: int,
    seed: int,
    with_equation_metric: bool,
) -> dict[str, Any]:
    print(f"    {name} ({len(problems)} items) [{decode['label']}] ...", end=" ", flush=True)
    prompts = [f"Question: {p['question'].strip()}\n" for p in problems]
    golds = [str(p["answer"]).strip() for p in problems]

    answer_close = tokenizer.encode("</answer>", add_bos=False, add_eos=False)
    stop_ids = {tokenizer.eos_id}
    if len(answer_close) == 1:
        stop_ids.add(int(answer_close[0]))

    gen_state = None
    if decode["temperature"] > 1e-6:
        gen_state = torch.Generator(device=device)
        gen_state.manual_seed(seed)

    all_gens: list[str] = []
    t0 = time.time()
    for i in range(0, len(prompts), batch_size):
        all_gens.extend(
            batch_generate_decoded(
                model, tokenizer, prompts[i : i + batch_size], device,
                max_new_tokens=256,
                temperature=decode["temperature"],
                top_p=decode["top_p"],
                repetition_penalty=decode["repetition_penalty"],
                generator=gen_state,
                stop_ids=stop_ids,
            )
        )
    elapsed = time.time() - t0

    n = len(problems)
    correct = 0
    full_format = 0
    think_ok = 0
    answer_ok = 0
    eos_ok = 0
    rep = 0
    lengths: list[int] = []
    extracted_all: list = []
    buckets = {"2_step": [0, 0], "3_step": [0, 0], "4plus_step": [0, 0]}
    samples: list[dict[str, Any]] = []

    for idx, (p, gen, gold) in enumerate(zip(problems, all_gens, golds)):
        toks = tokenizer.encode(gen, add_bos=False, add_eos=False)
        lengths.append(len(toks))
        # Task 21B definitions, verbatim (evaluate_task21b_transfer.py:206-211)
        if ("<think>" in gen and "</think>" in gen) or ("</think>" in gen):
            think_ok += 1
        if ("<answer>" in gen and "</answer>" in gen) or (gen.startswith("<answer>") and "</answer>" in gen):
            answer_ok += 1
        if "<think>" in gen and "</think>" in gen and "<answer>" in gen and "</answer>" in gen:
            full_format += 1
        if "</answer>" in gen or len(toks) < 256:
            eos_ok += 1
        if check_repetition(toks, n=4):
            rep += 1

        ext = extract_strict_answer(gen, is_mcq=False)
        extracted_all.append(ext)
        is_corr = False
        if ext is not None:
            try:
                is_corr = abs(float(ext) - float(gold)) < 1e-4
            except ValueError:
                is_corr = ext == gold
        if is_corr:
            correct += 1

        ops = p.get("metadata", {}).get("num_arithmetic_operations", 2)
        key = "2_step" if ops <= 2 else ("3_step" if ops == 3 else "4plus_step")
        buckets[key][1] += 1
        if is_corr:
            buckets[key][0] += 1

        if idx < 8:
            samples.append({
                "id": p.get("id", f"{name}_{idx}"),
                "question": p["question"][:160],
                "gold_answer": gold,
                "extracted_answer": ext,
                "is_correct": is_corr,
                "generation": gen[:400],
            })

    res: dict[str, Any] = {
        "N": n,
        "correct": correct,
        "strict_accuracy": correct / n * 100.0 if n else 0.0,
        "think_compliance_rate": think_ok / n * 100.0 if n else 0.0,
        "answer_compliance_rate": answer_ok / n * 100.0 if n else 0.0,
        "full_format_compliance_rate": full_format / n * 100.0 if n else 0.0,
        "eos_rate": eos_ok / n * 100.0 if n else 0.0,
        "repetition_rate": rep / n * 100.0 if n else 0.0,
        "generation_length": {
            "mean": float(np.mean(lengths)),
            "p50": float(np.median(lengths)),
            "p90": float(np.percentile(lengths, 90)),
        },
        "complexity_stratification": {
            k: {"N": t, "correct": c, "accuracy": (c / t * 100.0) if t else 0.0}
            for k, (c, t) in buckets.items()
        },
        "samples": samples,
        "eval_time_sec": elapsed,
    }
    if with_equation_metric:
        res["equation_metrics"] = equation_validity(all_gens, extracted_all)
        print(f"acc {res['strict_accuracy']:.2f}% | eq-validity "
              f"{res['equation_metrics']['step_level_equation_validity']:.2f}% "
              f"({res['equation_metrics']['valid_generated_equations']}/"
              f"{res['equation_metrics']['total_generated_equations']}) | {elapsed:.0f}s")
    else:
        print(f"acc {res['strict_accuracy']:.2f}% | {elapsed:.0f}s")
    return res


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #
def subtraction_holdout(lo: int, hi: int, count: int = 500, seed: int = 42) -> list[dict[str, Any]]:
    """Identical construction to the Task 21B retention probe (same seed)."""
    rng = np.random.default_rng(seed)
    items = []
    for i in range(count):
        a = int(rng.integers(lo, hi + 1))
        b = int(rng.integers(lo, a + 1))
        items.append({
            "id": f"sub_{lo}_{hi}_{i}",
            "question": f"Calculate: {a} - {b}",
            "answer": str(a - b),
            "metadata": {"num_arithmetic_operations": 1},
        })
    return items


def load_replay_holdout(path: str) -> list[dict[str, Any]]:
    """Held-out operator-matched arithmetic (the regime the replay corpus targets)."""
    items = []
    if not os.path.exists(path):
        return items
    for i, line in enumerate(open(path, encoding="utf-8")):
        if not line.strip():
            continue
        r = json.loads(line)
        items.append({
            "id": r["id"],
            "question": r["question"],
            "answer": r["answer"],
            "metadata": {"num_arithmetic_operations": r["metadata"]["num_steps"]},
        })
    return items


def load_gsm8k_test() -> list[dict[str, Any]]:
    mgr = ReasoningDatasetManager(seed=42)
    raw = mgr.get_split_raw("gsm8k", "test")
    out = []
    for idx, ex in enumerate(raw):
        ans_raw = ex.get("answer", "")
        gold = ans_raw.split("####")[-1].strip() if "####" in ans_raw else ans_raw.strip()
        gold = re.sub(r"[^\d\.\-]", "", gold)
        n_calc = len(re.findall(r"<<.*?>>", ans_raw))
        reasoning = ans_raw.split("####")[0] if "####" in ans_raw else ans_raw
        sents = [s.strip() for s in re.split(r"\n+|\.\s+", reasoning) if s.strip()]
        out.append({
            "id": f"gsm8k_test_{idx}",
            "question": ex["question"].strip(),
            "answer": gold,
            "metadata": {"num_arithmetic_operations": max(1, n_calc, len(sents))},
        })
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Task 22 evaluation")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gsm8k_limit", type=int, default=0, help="0 = full test set")
    ap.add_argument("--skip_ppl", action="store_true")
    ap.add_argument("--out_json", default="reports/task22_replay_sft.json")
    ap.add_argument("--partial_json", default="reports/task22_replay_sft.partial.json",
                    help="Per-model incremental results; used to resume after an interruption.")
    ap.add_argument("--force", action="store_true", help="Ignore the partial file and re-evaluate everything.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("MINI-R1 -- TASK 22: OPERATOR-MATCHED REPLAY SFT EVALUATION")
    print(f"Device: {device} | batch_size={args.batch_size} | seed={args.seed}")
    print("=" * 80)

    tok_path = "artifacts/tokenizer/tokenizer.model"
    tokenizer = Tokenizer(tok_path)

    print("\n[1/3] Loading benchmarks and probes ...")
    gsm8k_test = load_gsm8k_test()
    if args.gsm8k_limit:
        gsm8k_test = gsm8k_test[: args.gsm8k_limit]
    ret_2d = subtraction_holdout(50, 99, 500, seed=42)
    ret_3d = subtraction_holdout(101, 199, 500, seed=42)
    ret_4d = subtraction_holdout(1001, 1999, 500, seed=42)
    replay_holdout = load_replay_holdout("data/replay_arithmetic/dev.jsonl")
    print(f"  GSM8K test={len(gsm8k_test)} | retention 2D/3D/4D=500 each | "
          f"replay holdout={len(replay_holdout)}")

    models = [
        {
            "id": "Model_G_Task23_90_10",
            "label": "Model G (Task 23: 90/10 replay, all trainable)",
            "path": "artifacts/sft/task23_90_10/best.pt",
        },
        {
            "id": "Model_H_Task23_75_25",
            "label": "Model H (Task 23: 75/25 replay, all trainable)",
            "path": "artifacts/sft/task23_75_25/best.pt",
        },
        {
            "id": "Model_F_Task22B_Replay_All_Trainable",
            "label": "Model F (Task 22B: Replay, All Params Trainable)",
            "path": "artifacts/sft/task22b_replay_only/best.pt",
        },
        {
            "id": "Model_E_Task22_Replay_Isolated",
            "label": "Model E (Task 22: Replay + Parameter Isolation)",
            "path": "artifacts/sft/task22_replay_isolated/best.pt",
        },
        {
            "id": "Model_A_21B_Compact_Transfer",
            "label": "Model A (21B: Arithmetic -> Compact GSM8K)",
            "path": "artifacts/sft/compact_transfer/arithmetic_init/best.pt",
        },
        {
            "id": "Model_B_21B_Compact_Transfer",
            "label": "Model B (21B: Base -> Compact GSM8K)",
            "path": "artifacts/sft/compact_transfer/base_init/best.pt",
        },
    ]

    results: dict[str, Any] = {}
    if os.path.exists(args.partial_json) and not args.force:
        with open(args.partial_json, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Resuming: {args.partial_json} already holds {sorted(results)}")
    print("\n[2/3] Evaluating models ...")

    for spec in models:
        path = spec["path"]
        if spec["id"] in results:
            print(f"\n>>> {spec['label']} -- already in partial results, skipping")
            continue
        if not os.path.exists(path):
            print(f"\n[WARNING] missing checkpoint {path} -- skipping {spec['label']}")
            continue
        print(f"\n>>> {spec['label']}")
        print(f"    {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = MiniR1Config.from_dict(ckpt.get("model_config", ckpt.get("config", {})))
        model = MiniR1Transformer(cfg).to(device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model", {})))
        model.eval()

        entry: dict[str, Any] = {
            "model_label": spec["label"],
            "checkpoint_path": path,
            "checkpoint_sha256": compute_sha256(path),
            "gsm8k_by_decode": {},
        }

        # GSM8K under both decoding conditions (equation metric on both)
        for dk, dcfg in DECODE_CONFIGS.items():
            entry["gsm8k_by_decode"][dk] = run_benchmark(
                model, tokenizer, gsm8k_test, "GSM8K_test", device, dcfg,
                args.batch_size, args.seed, with_equation_metric=True,
            )

        # Retention probes: greedy only, so they line up with the Task 20
        # pre-transfer baseline and the Task 21B reported retention numbers.
        greedy = DECODE_CONFIGS["greedy"]
        ret = {}
        for tag, probs in (("2D_subtraction", ret_2d), ("3D_subtraction", ret_3d), ("4D_subtraction", ret_4d)):
            r = run_benchmark(model, tokenizer, probs, f"Retention_{tag}", device, greedy,
                              args.batch_size, args.seed, with_equation_metric=False)
            ret[tag] = {"N": r["N"], "correct": r["correct"], "accuracy": r["strict_accuracy"]}
        entry["arithmetic_retention"] = ret

        if replay_holdout:
            r = run_benchmark(model, tokenizer, replay_holdout, "Replay_regime_holdout", device, greedy,
                              args.batch_size, args.seed, with_equation_metric=True)
            entry["replay_regime_holdout"] = {
                "N": r["N"], "correct": r["correct"], "accuracy": r["strict_accuracy"],
                "step_level_equation_validity": r["equation_metrics"]["step_level_equation_validity"],
                "total_generated_equations": r["equation_metrics"]["total_generated_equations"],
            }

        if not args.skip_ppl:
            print("    General-language PPL ...", end=" ", flush=True)
            g = evaluate_general_language(model, "data/pretrain/val_manifest.json",
                                          batch_size=32, device=device, use_amp=True)
            entry["general_language"] = {"val_loss": g.avg_loss, "perplexity": g.perplexity}
            print(f"ppl {g.perplexity:.2f}")

        results[spec["id"]] = entry
        os.makedirs("reports", exist_ok=True)
        with open(args.partial_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"    [checkpointed -> {args.partial_json}]")
        del model
        torch.cuda.empty_cache()

    # ---- deltas -----------------------------------------------------------
    print("\n[3/3] Writing report ...")

    def eqv(mid: str, dk: str) -> float:
        return (results.get(mid, {}).get("gsm8k_by_decode", {}).get(dk, {})
                .get("equation_metrics", {}).get("step_level_equation_validity", 0.0))

    def acc(mid: str, dk: str) -> float:
        return results.get(mid, {}).get("gsm8k_by_decode", {}).get(dk, {}).get("strict_accuracy", 0.0)

    E, A, B = "Model_E_Task22_Replay_Isolated", "Model_A_21B_Compact_Transfer", "Model_B_21B_Compact_Transfer"
    Fm = "Model_F_Task22B_Replay_All_Trainable"
    primary = "sampled_t06_p095_rp12"
    deltas = {
        "primary_decode": primary,
        "equation_validity": {
            "E_minus_A_primary": eqv(E, primary) - eqv(A, primary),
            "E_minus_B_primary": eqv(E, primary) - eqv(B, primary),
            "E_minus_A_greedy": eqv(E, "greedy") - eqv(A, "greedy"),
            "E_minus_B_greedy": eqv(E, "greedy") - eqv(B, "greedy"),
        },
        "delta_replay_trainability": {
            "_definition": "Model F (replay, all trainable) minus Model E (replay, frozen). Isolates parameter freezing from replay data.",
            "equation_validity_primary": eqv(Fm, primary) - eqv(E, primary),
            "equation_validity_greedy": eqv(Fm, "greedy") - eqv(E, "greedy"),
            "gsm8k_acc_primary": acc(Fm, primary) - acc(E, primary),
            "gsm8k_acc_greedy": acc(Fm, "greedy") - acc(E, "greedy"),
        },
        "task22b_vs_21b_baselines": {
            "F_minus_A_primary_validity": eqv(Fm, primary) - eqv(A, primary),
            "F_minus_B_primary_validity": eqv(Fm, primary) - eqv(B, primary),
        },
        "gsm8k_strict_accuracy": {
            "E_minus_A_primary": acc(E, primary) - acc(A, primary),
            "E_minus_B_primary": acc(E, primary) - acc(B, primary),
            "E_minus_A_greedy": acc(E, "greedy") - acc(A, "greedy"),
            "E_minus_B_greedy": acc(E, "greedy") - acc(B, "greedy"),
        },
    }

    d = deltas["equation_validity"]["E_minus_A_primary"]
    if d >= 10.0:
        verdict = f"Conclusion A: Step-level equation validity rises materially (+{d:.2f} pts vs Task 21B Model A). SUCCESS CRITERION MET."
    elif d >= 3.0:
        verdict = f"Conclusion B: Step-level equation validity rises modestly (+{d:.2f} pts vs Task 21B Model A)."
    elif abs(d) < 3.0:
        verdict = f"Conclusion C: Step-level equation validity is flat ({d:+.2f} pts vs Task 21B Model A). SUCCESS CRITERION NOT MET."
    else:
        verdict = f"Conclusion D: Step-level equation validity regressed ({d:+.2f} pts vs Task 21B Model A)."

    report = {
        "metadata": {
            "task": "Task 22: Operator-Matched Replay SFT with Parameter Isolation",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "decode_configs": DECODE_CONFIGS,
            "primary_metric": "step_level_equation_validity",
            "generation_contract": (
                "Task 21B batching contract preserved verbatim: BOS-prefixed prompt, "
                "left padding, no attention mask passed to the model, max_new_tokens=256."
            ),
            "retention_probe_decoding": "greedy (matches Task 20 baseline and Task 21B report)",
            "generation_stop_rule": (
                "Generation halts on EOS or on the </answer> token (a single dedicated token id). "
                "Equation validity, strict accuracy and full-format compliance are byte-identical to "
                "running to 256 tokens, because all three read only the first <think>..</think> and the "
                "first <answer>..</answer>. Reported generation LENGTH and REPETITION are therefore "
                "measured up to and including </answer>, and are NOT comparable to the Task 21B report's "
                "length/repetition figures, which included post-answer continuation."
            ),
            "deltas": deltas,
            "verdict": verdict,
        },
        "models": results,
    }
    os.makedirs("reports", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ---- console summary --------------------------------------------------
    print("\n" + "=" * 100)
    print("STEP-LEVEL EQUATION VALIDITY (primary metric) -- GSM8K test generations")
    print("=" * 100)
    hdr = f"{'Model':<48} {'decode':<24} {'valid/total':>16} {'validity':>10} {'GSM8K acc':>10}"
    print(hdr)
    print("-" * 100)
    for mid, m in results.items():
        for dk in DECODE_CONFIGS:
            g = m["gsm8k_by_decode"].get(dk)
            if not g:
                continue
            em = g["equation_metrics"]
            print(f"{m['model_label'][:47]:<48} {dk:<24} "
                  f"{em['valid_generated_equations']:>7}/{em['total_generated_equations']:<8} "
                  f"{em['step_level_equation_validity']:>9.2f}% {g['strict_accuracy']:>9.2f}%")
    print("-" * 100)
    print(f"\nPrimary delta (Model E - Model A, {primary}): "
          f"{deltas['equation_validity']['E_minus_A_primary']:+.2f} pts")
    print(f"Primary delta (Model E - Model B, {primary}): "
          f"{deltas['equation_validity']['E_minus_B_primary']:+.2f} pts")
    print(f"\nVERDICT: {verdict}")
    print(f"\nSaved {args.out_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
