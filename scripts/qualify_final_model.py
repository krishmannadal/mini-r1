"""
qualify_final_model.py - Task 24: final qualification of the submission model.

Evaluates ONLY artifacts/sft/task23_75_25/best.pt.

Reuses validated primitives verbatim:
  - batch_generate_decoded, equation_validity, DECODE_CONFIGS  (Task 22 evaluator)
  - extract_strict_answer, check_repetition                    (Task 21B evaluator)
  - evaluate_general_language, ReasoningDatasetManager         (src)
Neither existing evaluator is modified.

Adds only the benchmark coverage Task 24 requires that no single existing
script had in one place: SVAMP, AQuA, and the Task 20 E1-E7 arithmetic suites
(IID / OOD / 5-digit acid test).

GSM8K, the 2D/3D/4D retention probes and general-language PPL are read from the
already-computed reports/task23_replay_ratio.json rather than regenerated, so
the numbers are bit-identical to the ones the model was selected on.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
for _p in (PROJECT_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluate_task22_replay import (  # noqa: E402
    DECODE_CONFIGS,
    batch_generate_decoded,
    equation_validity,
)
from evaluate_task21b_transfer import check_repetition, extract_strict_answer  # noqa: E402
from src.datasets import ReasoningDatasetManager  # noqa: E402
from src.evaluation import evaluate_general_language  # noqa: E402
from src.model import MiniR1Config, MiniR1Transformer  # noqa: E402
from src.tokenizer import Tokenizer  # noqa: E402

FINAL_CKPT = "artifacts/sft/task23_75_25/best.pt"
TOKENIZER = "artifacts/tokenizer/tokenizer.model"
CACHED = "reports/task23_replay_ratio.json"
CACHED_ID = "Model_H_Task23_75_25"

E_SUITES = [
    ("E1_iid_2digit", "data/progressive_subtraction_sft/eval_e1_iid_2digit.json", "IID 2D [50,99]"),
    ("E2_ood_2digit", "data/progressive_subtraction_sft/eval_e2_ood_2digit.json", "OOD 2D"),
    ("E3_iid_3digit", "data/progressive_subtraction_sft/eval_e3_iid_3digit.json", "IID 3D [101,199]"),
    ("E4_ood_3digit", "data/progressive_subtraction_sft/eval_e4_ood_3digit.json", "OOD 3D"),
    ("E5_iid_4digit", "data/progressive_subtraction_sft/eval_e5_iid_4digit.json", "IID 4D [1001,1999]"),
    ("E6_ood_4digit", "data/progressive_subtraction_sft/eval_e6_ood_4digit.json", "OOD 4D"),
    ("E7_acid_5digit", "data/progressive_subtraction_sft/eval_e7_acid_5digit.json", "ACID 5D (never trained)"),
]


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def run_suite(model, tokenizer, problems, name, device, decode, batch_size, seed, is_mcq=False):
    """Same measurement contract as the Task 22 evaluator, with MCQ support."""
    print(f"  {name} ({len(problems)}) [{decode['label']}] ...", end=" ", flush=True)
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

    gens = []
    t0 = time.time()
    for i in range(0, len(prompts), batch_size):
        gens.extend(batch_generate_decoded(
            model, tokenizer, prompts[i:i + batch_size], device, max_new_tokens=256,
            temperature=decode["temperature"], top_p=decode["top_p"],
            repetition_penalty=decode["repetition_penalty"],
            generator=gen_state, stop_ids=stop_ids,
        ))
    elapsed = time.time() - t0

    n = len(problems)
    correct = think = ans_ok = full = eos = rep = 0
    lengths, extracted = [], []
    for p, gen, gold in zip(problems, gens, golds):
        toks = tokenizer.encode(gen, add_bos=False, add_eos=False)
        lengths.append(len(toks))
        if ("<think>" in gen and "</think>" in gen) or ("</think>" in gen):
            think += 1
        if ("<answer>" in gen and "</answer>" in gen) or (gen.startswith("<answer>") and "</answer>" in gen):
            ans_ok += 1
        if "<think>" in gen and "</think>" in gen and "<answer>" in gen and "</answer>" in gen:
            full += 1
        if "</answer>" in gen or len(toks) < 256:
            eos += 1
        if check_repetition(toks, n=4):
            rep += 1
        ext = extract_strict_answer(gen, is_mcq=is_mcq)
        extracted.append(ext)
        if ext is not None:
            if is_mcq:
                correct += int(ext.upper() == gold.upper())
            else:
                try:
                    correct += int(abs(float(ext) - float(gold)) < 1e-4)
                except ValueError:
                    correct += int(ext == gold)

    em = equation_validity(gens, extracted)
    res = {
        "N": n, "correct": correct,
        "strict_accuracy": correct / n * 100.0,
        "think_compliance_rate": think / n * 100.0,
        "answer_compliance_rate": ans_ok / n * 100.0,
        "full_format_compliance_rate": full / n * 100.0,
        "eos_rate": eos / n * 100.0,
        "repetition_rate": rep / n * 100.0,
        "generation_length": {"mean": float(np.mean(lengths)),
                              "p50": float(np.median(lengths)),
                              "p90": float(np.percentile(lengths, 90))},
        "equation_metrics": em,
        "eval_time_sec": elapsed,
        "decode": decode["label"],
        "samples": [{"question": p["question"][:140], "gold": g, "extracted": e, "generation": gn[:220]}
                    for p, g, e, gn in list(zip(problems, golds, extracted, gens))[:5]],
    }
    print(f"acc {res['strict_accuracy']:.2f}% | eqV {em['step_level_equation_validity']:.2f}% "
          f"| EOS {res['eos_rate']:.1f}% | {elapsed:.0f}s")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="reports/task24_final_benchmarks.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print("MINI-R1 -- TASK 24: FINAL MODEL QUALIFICATION")
    print(f"Checkpoint: {FINAL_CKPT}")
    print("=" * 78)

    tokenizer = Tokenizer(TOKENIZER)
    ck = torch.load(FINAL_CKPT, map_location="cpu", weights_only=False)
    cfg = MiniR1Config.from_dict(ck["model_config"])
    model = MiniR1Transformer(cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded. params={n_params:,} | ckpt sha={sha256_of(FINAL_CKPT)[:16]}...")

    greedy = DECODE_CONFIGS["greedy"]
    out = {
        "model": {
            "checkpoint": FINAL_CKPT,
            "checkpoint_sha256": sha256_of(FINAL_CKPT),
            "tokenizer": TOKENIZER,
            "tokenizer_sha256": sha256_of(TOKENIZER),
            "param_count": n_params,
            "model_config": ck["model_config"],
            "training_step": ck["step"],
            "label": ck.get("model_name"),
        },
        "benchmarks": {},
    }

    # ---- SVAMP / AQuA (not covered by the Task 22 evaluator) --------------
    print("\n[1/3] SVAMP + AQuA")
    mgr = ReasoningDatasetManager(seed=42)
    svamp = []
    for i, ex in enumerate(mgr.get_split_raw("svamp", "test")):
        body, q = ex.get("Body", "").strip(), ex.get("Question", "").strip()
        a = str(ex.get("Answer", "")).strip()
        try:
            f = float(a)
            a = str(int(f)) if f.is_integer() else f"{f:.2f}"
        except ValueError:
            pass
        svamp.append({"id": str(ex.get("ID", i)), "question": (f"{body} {q}".strip() if body else q), "answer": a})
    aqua = []
    for i, ex in enumerate(mgr.get_split_raw("aqua", "test")):
        opts = ex.get("options", [])
        opt = "\nOptions:\n" + "\n".join(f"- {o}" for o in opts) if opts else ""
        aqua.append({"id": f"aqua_{i}", "question": f"{ex.get('question','').strip()}{opt}".strip(),
                     "answer": str(ex.get("correct", "")).strip().upper()})
    out["benchmarks"]["svamp_test"] = run_suite(model, tokenizer, svamp, "SVAMP_test", device, greedy, args.batch_size, args.seed)
    out["benchmarks"]["aqua_test"] = run_suite(model, tokenizer, aqua, "AQuA_test", device, greedy, args.batch_size, args.seed, is_mcq=True)

    # ---- E1-E7 arithmetic suites (IID / OOD / acid) ----------------------
    print("\n[2/3] Task 20 arithmetic suites E1-E7")
    for key, path, desc in E_SUITES:
        raw = json.load(open(path, encoding="utf-8"))
        probs = [{"id": r["example_id"], "question": r["prompt"].replace("Question: ", "").strip(),
                  "answer": str(r["final_answer"])} for r in raw]
        r = run_suite(model, tokenizer, probs, key, device, greedy, args.batch_size, args.seed)
        r["description"] = desc
        r["source_file"] = path
        out["benchmarks"][key] = r

    # ---- carry forward the already-computed selection metrics ------------
    print("\n[3/3] Carrying forward cached GSM8K / retention / PPL (identical run)")
    cached = json.load(open(CACHED, encoding="utf-8"))["models"][CACHED_ID]
    assert cached["checkpoint_sha256"] == out["model"]["checkpoint_sha256"], \
        "cached results are for a different checkpoint"
    out["benchmarks"]["gsm8k_test"] = cached["gsm8k_by_decode"]
    out["benchmarks"]["arithmetic_retention_probe"] = cached["arithmetic_retention"]
    out["benchmarks"]["replay_regime_holdout"] = cached.get("replay_regime_holdout")
    out["general_language"] = cached["general_language"]
    print(f"  GSM8K greedy acc {cached['gsm8k_by_decode']['greedy']['strict_accuracy']:.2f}% | "
          f"PPL {cached['general_language']['perplexity']:.2f}  [cache SHA verified]")

    os.makedirs("reports", exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nSaved {args.out}")
    print("=" * 78)


if __name__ == "__main__":
    main()
