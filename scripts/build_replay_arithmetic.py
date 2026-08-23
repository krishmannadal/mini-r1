"""
build_replay_arithmetic.py - Task 22: Operator-Matched Replay Arithmetic Corpus.

Motivation
----------
Task 20 trained SUBTRACTION ONLY over [50,99] u [101,199] u [1001,1999].
The compact GSM8K traces (Task 21B) are dominated by '*' and '+' over small
operands (Step 1 measurement: * 37.40%, + 28.97%, / 17.13%, - 16.50%; operands
40.85% in 1-9 and 39.60% in 10-99). The arithmetic substrate therefore never
covered GSM8K's operator/range distribution, which is why Task 21B measured no
transfer.

This script generates a REPLAY corpus whose operator, operand-magnitude,
arity, step-count and result-carry statistics MATCH the empirical compact-GSM8K
trace distribution rather than the Task 18A/20 ranges.

Method
------
1. Measure the target distribution directly from
   data/gsm8k_compact_reasoning/train.jsonl (no hard-coded constants).
2. Generate traces as expression TREES (a pool of at most 2 open sub-results,
   collapsed to a single root), so every trace has one well-defined prompt
   expression and one verifiable final answer.
3. Fit the free sampling weights (fresh-operator weights, fresh-operand-bucket
   weights) by iterative proportional fitting so the REALIZED corpus statistics
   match the measured GSM8K targets.
4. Verify every equation and every prompt expression with deterministic Python
   arithmetic. Any trace with a single failing check is discarded.

Output (same compact schema as data/gsm8k_compact_reasoning/):
    data/replay_arithmetic/train.jsonl
    data/replay_arithmetic/dev.jsonl
    data/replay_arithmetic/manifest.json
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BUCKETS = ["0-<1", "1-9", "10-99", "100-999", "1000+"]
BUCKET_RANGE = {
    "1-9": (1, 9),
    "10-99": (10, 99),
    "100-999": (100, 999),
    "1000+": (1000, 9999),
}
OPS = ["+", "-", "*", "/"]
PREC = {"+": 1, "-": 1, "*": 2, "/": 2}
TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[+\-*/()]")
SAFE_EXPR = re.compile(r"^[\d\.\s\+\-\*\/\(\)]+$")


# --------------------------------------------------------------------------- #
# Target measurement
# --------------------------------------------------------------------------- #
def bucket_of(v: float) -> str:
    a = abs(float(v))
    if a < 1:
        return "0-<1"
    if a < 10:
        return "1-9"
    if a < 100:
        return "10-99"
    if a < 1000:
        return "100-999"
    return "1000+"


def measure_targets(path: str) -> dict[str, Any]:
    """Measure the empirical compact-GSM8K trace distribution."""
    recs = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]

    op_c: Counter = Counter()
    operand_c: Counter = Counter()
    operand_pool: dict = {b: [] for b in BUCKETS}
    result_c: Counter = Counter()
    arity_c: Counter = Counter()
    steps_c: Counter = Counter()
    carry_hits = 0
    carry_slots = 0
    carry_left = 0

    for rec in recs:
        steps = rec["metadata"]["steps"]
        steps_c[len(steps)] += 1
        prev = None
        for eq in steps:
            lhs, rhs = eq.split("=")
            toks = TOKEN_RE.findall(lhs)
            ops = [t for t in toks if t in "+-*/"]
            nums = [t for t in toks if t not in "+-*/()"]
            op_c.update(ops)
            arity_c[len(ops)] += 1
            for n in nums:
                b = bucket_of(n)
                operand_c[b] += 1
                fv = float(n)
                # cap the 1000+ pool: a handful of GSM8K operands reach 1e9 and
                # would dominate the generated corpus with unrealistic magnitudes
                if b != "1000+" or fv <= 99999:
                    operand_pool[b].append(fv)
            result_c[bucket_of(float(rhs.strip()))] += 1
            if prev is not None and nums:
                carry_slots += 1
                if nums[0] == prev:
                    carry_hits += 1
                    carry_left += 1
                elif prev in nums[1:]:
                    carry_hits += 1
            prev = rhs.strip()

    def norm(counter: Counter, keys) -> dict:
        tot = sum(counter.get(k, 0) for k in keys) or 1
        return {k: counter.get(k, 0) / tot for k in keys}

    # arity: drop the degenerate 0-op rows, renormalise over arity 1..3
    arity_keys = [1, 2, 3]
    arity_tot = sum(arity_c.get(k, 0) for k in arity_keys) or 1
    arity_p = {k: arity_c.get(k, 0) / arity_tot for k in arity_keys}

    step_keys = sorted(k for k in steps_c if k >= 1)
    step_tot = sum(steps_c[k] for k in step_keys) or 1
    steps_p = {k: steps_c[k] / step_tot for k in step_keys}

    return {
        "n_traces": len(recs),
        "n_equations": sum(arity_c.values()),
        "operator_p": norm(op_c, OPS),
        "operand_bucket_p": norm(operand_c, BUCKETS),
        "operand_pool": operand_pool,
        "result_bucket_p": norm(result_c, BUCKETS),
        "arity_p": arity_p,
        "steps_per_trace_p": steps_p,
        "carry_rate": carry_hits / max(1, carry_slots),
        "carry_left_frac": carry_left / max(1, carry_hits),
    }


# --------------------------------------------------------------------------- #
# Expression nodes
# --------------------------------------------------------------------------- #
class Node:
    """An expression fragment with its verified value and lowest operator precedence."""

    __slots__ = ("expr", "value", "prec")

    def __init__(self, expr: str, value: float, prec: int):
        self.expr = expr
        self.value = value
        self.prec = prec


def fmt_num(v: float) -> str:
    """Format like the compact GSM8K builder: integer when whole, else <=2dp."""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def leaf(v: float) -> Node:
    return Node(fmt_num(v), float(v), 3)


def wrap(child: Node, op: str, is_right: bool) -> str:
    """Minimal parenthesisation under standard operator precedence."""
    if child.prec < PREC[op]:
        return "(" + child.expr + ")"
    if is_right and child.prec == PREC[op] and op in ("-", "/"):
        return "(" + child.expr + ")"
    return child.expr


def combine(a: Node, op: str, b: Node) -> Node:
    expr = wrap(a, op, False) + " " + op + " " + wrap(b, op, True)
    if op == "+":
        val = a.value + b.value
    elif op == "-":
        val = a.value - b.value
    elif op == "*":
        val = a.value * b.value
    else:
        val = a.value / b.value
    return Node(expr, val, PREC[op])


class Pair:
    """
    Two parallel renderings of the same arithmetic value.

    `step` is what the reasoning step prints: a previously computed sub-result
    appears as its NUMERIC VALUE, exactly as GSM8K traces do ('316 / 79 = 4').
    `full` is the fully nested expression used to build the prompt.
    """

    __slots__ = ("step", "full")

    def __init__(self, step: Node, full: Node):
        self.step = step
        self.full = full

    @property
    def value(self) -> float:
        return self.full.value


def fresh_pair(v: float) -> Pair:
    return Pair(leaf(v), leaf(v))


def carried_pair(node: Node) -> Pair:
    return Pair(leaf(node.value), node)


def combine_pair(a: Pair, op: str, b: Pair) -> Pair:
    return Pair(combine(a.step, op, b.step), combine(a.full, op, b.full))


# --------------------------------------------------------------------------- #
# Operand / step sampling
# --------------------------------------------------------------------------- #
def weighted_choice(rng: random.Random, weights: dict) -> Any:
    keys = list(weights.keys())
    tot = sum(weights[k] for k in keys)
    x = rng.random() * tot
    acc = 0.0
    for k in keys:
        acc += weights[k]
        if x <= acc:
            return k
    return keys[-1]


def sample_operand(rng: random.Random, bucket_w: dict, pool: dict) -> float:
    """Draw a fresh operand: pick a magnitude bucket by fitted weight, then draw
    a value from GSM8K's empirical within-bucket histogram (so round-number bias
    and the frequency of small operands such as 1 and 2 are reproduced)."""
    b = weighted_choice(rng, bucket_w)
    vals = pool.get(b)
    if vals:
        return rng.choice(vals)
    lo, hi = BUCKET_RANGE.get(b, (1, 9))
    return float(rng.randint(lo, hi))


def divisors_in_range(n: int, lo: int, hi: int, limit: int = 64) -> list:
    """Divisors of n inside [lo, hi]. 1 is excluded unless it is the only option
    (GSM8K uses '/ 1' in only 0.09% of equations)."""
    out: list = []
    i = 1
    while i * i <= n and len(out) < limit:
        if n % i == 0:
            for d in (i, n // i):
                if lo <= d <= hi and d not in out:
                    out.append(d)
        i += 1
    non_trivial = [d for d in out if d != 1]
    return non_trivial if non_trivial else out


def make_division(rng: random.Random, bucket_w: dict, pool: dict):
    """Construct (a, b) with a/b clean: exact integer, or <=2dp at a low rate."""
    nz = {k: v for k, v in bucket_w.items() if k != "0-<1"}
    bb = weighted_choice(rng, nz)
    ba = weighted_choice(rng, nz)
    b_lo, b_hi = BUCKET_RANGE[bb]
    a_lo, a_hi = BUCKET_RANGE[ba]

    if rng.random() < 0.033:  # decimal-quotient path (GSM8K: 3.32% of divisions)
        if b_hi < 20:
            return None
        start = max(20, (b_lo // 20) * 20)
        if start > b_hi:
            return None
        b = rng.randrange(start, b_hi + 1, 20)
        q = rng.randint(1, 40) + rng.choice([0.25, 0.5, 0.75, 0.2, 0.4, 0.6, 0.8])
        a = b * q
        if abs(a - round(a)) > 1e-9 or not (a_lo <= a <= a_hi):
            return None
        return float(round(a)), float(b)

    for _ in range(12):
        b = rng.randint(max(b_lo, 2), b_hi)
        q_lo = max(2, math.ceil(a_lo / b))
        q_hi = math.floor(a_hi / b)
        if q_hi < q_lo:
            continue
        q = rng.randint(q_lo, q_hi)
        return float(b * q), float(b)
    return None


def build_step(
    rng: random.Random,
    pool_items: list,
    fresh_needed: int,
    op_w: dict,
    bucket_w: dict,
    pool: dict,
    carry_left_frac: float,
):
    """
    Build one reasoning step consuming `pool_items` (0-2 open sub-results) plus
    `fresh_needed` freshly sampled operands. Returns a Pair or None.
    """
    n_ops = len(pool_items) + fresh_needed - 1
    if n_ops < 1:
        return None

    # ---- single-operator step: '/' handled constructively for exactness ----
    if n_ops == 1:
        op = weighted_choice(rng, op_w)

        if len(pool_items) == 0:
            if op == "/":
                div = make_division(rng, bucket_w, pool)
                if div is None:
                    return None
                a, b = div
                return combine_pair(fresh_pair(a), "/", fresh_pair(b))
            a = sample_operand(rng, bucket_w, pool)
            b = sample_operand(rng, bucket_w, pool)
            if op == "-" and b > a:
                a, b = b, a
            return combine_pair(fresh_pair(a), op, fresh_pair(b))

        if len(pool_items) == 2:
            x, y = carried_pair(pool_items[0]), carried_pair(pool_items[1])
            if rng.random() < 0.5:
                x, y = y, x
            if op == "/":
                if abs(y.value) < 1e-9 or abs(x.value % y.value) > 1e-9:
                    return None
            if op == "-" and y.value > x.value:
                x, y = y, x
            return combine_pair(x, op, y)

        # exactly one carried sub-result + one fresh operand
        carried = carried_pair(pool_items[0])
        if op == "/":
            iv = carried.value
            if abs(iv - round(iv)) > 1e-9 or round(iv) <= 0:
                return None
            nz = {k: v for k, v in bucket_w.items() if k != "0-<1"}
            lo, hi = BUCKET_RANGE[weighted_choice(rng, nz)]
            divs = divisors_in_range(int(round(iv)), lo, hi)
            if not divs:
                return None
            return combine_pair(carried, "/", fresh_pair(float(rng.choice(divs))))

        fresh = fresh_pair(sample_operand(rng, bucket_w, pool))
        if op == "-":
            hi_p, lo_p = (carried, fresh) if carried.value >= fresh.value else (fresh, carried)
            return combine_pair(hi_p, "-", lo_p)
        if rng.random() < carry_left_frac:
            return combine_pair(carried, op, fresh)
        return combine_pair(fresh, op, carried)

    # ---- multi-operator step (arity 2-3): flat chain, no division ----------
    no_div = {k: v for k, v in op_w.items() if k != "/"}
    ops_seq = [weighted_choice(rng, no_div) for _ in range(n_ops)]
    operands = [carried_pair(p) for p in pool_items]
    operands += [fresh_pair(sample_operand(rng, bucket_w, pool)) for _ in range(fresh_needed)]
    rng.shuffle(operands)

    node = operands[0]
    for i, o in enumerate(ops_seq):
        nxt = operands[i + 1]
        if o == "-" and nxt.value > node.value:
            o = "+"
        node = combine_pair(node, o, nxt)
    return node


# --------------------------------------------------------------------------- #
# Trace generation
# --------------------------------------------------------------------------- #
def generate_trace(rng: random.Random, targets: dict, op_w: dict, bucket_w: dict, result_w: dict):
    n_steps = int(weighted_choice(rng, targets["steps_per_trace_p"]))
    carry_rate = targets["carry_rate"]
    carry_left_frac = targets["carry_left_frac"]
    arity_p = targets["arity_p"]
    operand_pool = targets["operand_pool"]

    open_pool: list = []
    steps: list = []

    for si in range(n_steps):
        remaining_after = n_steps - si - 1

        # How many open sub-results does this step consume?
        # Invariant: open_pool size never exceeds 2 and must reach 1 at the end.
        if si == 0:
            consume = 0
        elif len(open_pool) == 2:
            consume = 2 if (remaining_after == 0 or rng.random() < 0.5) else 1
        else:  # len(open_pool) == 1
            if remaining_after > 0 and rng.random() > carry_rate:
                consume = 0  # branch: open a second independent quantity
            else:
                consume = 1
        if remaining_after == 0:
            consume = len(open_pool)

        # Retry the step until its result magnitude passes the fitted
        # result-bucket acceptance filter (bounded).
        accepted = None
        for _ in range(24):
            arity = int(weighted_choice(rng, arity_p))
            fresh_needed = max(0, (arity + 1) - consume)
            if consume + fresh_needed < 2:
                fresh_needed = 2 - consume

            pair = build_step(rng, open_pool[:consume], fresh_needed, op_w, bucket_w, operand_pool, carry_left_frac)
            if pair is None:
                continue
            if pair.value < 0 or abs(pair.value) > 5_000_000:
                continue
            rv = round(pair.value, 2)
            if abs(pair.value - rv) > 1e-9:
                continue
            if rng.random() > result_w.get(bucket_of(rv), 1.0):
                continue
            accepted = (pair, rv)
            break

        if accepted is None:
            return None
        pair, rv = accepted

        steps.append(pair.step.expr + " = " + fmt_num(rv))
        open_pool = open_pool[consume:] + [Node(pair.full.expr, rv, pair.full.prec)]

    if len(open_pool) != 1:
        return None
    root = open_pool[0]
    return {"steps": steps, "expr": root.expr, "answer": fmt_num(round(root.value, 2))}


# --------------------------------------------------------------------------- #
# Deterministic verification
# --------------------------------------------------------------------------- #
def verify_trace(trace: dict):
    """Independently re-verify every equation and the prompt expression."""
    for eq in trace["steps"]:
        if eq.count("=") != 1:
            return False, "bad_equation_shape"
        lhs, rhs = eq.split("=")
        lhs, rhs = lhs.strip(), rhs.strip()
        if not SAFE_EXPR.match(lhs) or not SAFE_EXPR.match(rhs):
            return False, "unsafe_characters"
        try:
            lv = eval(lhs, {"__builtins__": None}, {})
            rv = float(rhs)
        except Exception:
            return False, "eval_error"
        if abs(lv - rv) > 1e-6:
            return False, "step_mismatch"
        if fmt_num(rv) != rhs:
            return False, "formatting_mismatch"

    expr = trace["expr"]
    if not SAFE_EXPR.match(expr):
        return False, "unsafe_prompt_expr"
    try:
        pv = eval(expr, {"__builtins__": None}, {})
    except Exception:
        return False, "prompt_eval_error"
    if abs(pv - float(trace["answer"])) > 1e-6:
        return False, "prompt_answer_mismatch"
    if trace["steps"][-1].split("=")[1].strip() != trace["answer"]:
        return False, "final_step_answer_mismatch"
    return True, "valid"


# --------------------------------------------------------------------------- #
# Realised statistics + iterative proportional fitting
# --------------------------------------------------------------------------- #
def measure_corpus(traces: list) -> dict:
    op_c: Counter = Counter()
    operand_c: Counter = Counter()
    result_c: Counter = Counter()
    arity_c: Counter = Counter()
    steps_c: Counter = Counter()

    for t in traces:
        steps_c[len(t["steps"])] += 1
        for eq in t["steps"]:
            lhs, rhs = eq.split("=")
            toks = TOKEN_RE.findall(lhs)
            op_c.update([x for x in toks if x in "+-*/"])
            arity_c[len([x for x in toks if x in "+-*/"])] += 1
            for n in [x for x in toks if x not in "+-*/()"]:
                operand_c[bucket_of(n)] += 1
            result_c[bucket_of(float(rhs.strip()))] += 1

    def norm(counter, keys):
        tot = sum(counter.get(k, 0) for k in keys) or 1
        return {k: counter.get(k, 0) / tot for k in keys}

    return {
        "operator_p": norm(op_c, OPS),
        "operand_bucket_p": norm(operand_c, BUCKETS),
        "result_bucket_p": norm(result_c, BUCKETS),
        "arity_p": norm(arity_c, [1, 2, 3]),
        "steps_per_trace_p": norm(steps_c, sorted(steps_c)),
        "n_equations": sum(arity_c.values()),
    }


def tvd(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def generate_pool(rng, n, targets, op_w, bucket_w, result_w, attempts_mult=60):
    out: list = []
    attempts = 0
    cap = n * attempts_mult
    while len(out) < n and attempts < cap:
        attempts += 1
        t = generate_trace(rng, targets, op_w, bucket_w, result_w)
        if t is None:
            continue
        ok, _ = verify_trace(t)
        if ok:
            out.append(t)
    return out


def fit_weights(targets: dict, seed: int, iters: int, probe_n: int):
    """
    Iterative proportional fitting of the free sampling knobs:
      op_w      - fresh operator weights
      bucket_w  - fresh operand magnitude-bucket weights
      result_w  - per-result-bucket acceptance probabilities (max 1.0)
    against the measured compact-GSM8K marginals.
    """
    op_w = dict(targets["operator_p"])
    bucket_w = dict(targets["operand_bucket_p"])
    result_w = {k: 1.0 for k in BUCKETS}

    print("")
    print("[IPF] Fitting sampling weights to the measured GSM8K targets")
    print("{:>4} {:>9} {:>13} {:>12} {:>8}".format("iter", "op TVD", "operand TVD", "result TVD", "n_eq"))
    for it in range(iters):
        rng = random.Random(seed + 1000 + it)
        pool = generate_pool(rng, probe_n, targets, op_w, bucket_w, result_w)
        if not pool:
            break
        m = measure_corpus(pool)
        d_op = tvd(m["operator_p"], targets["operator_p"])
        d_bk = tvd(m["operand_bucket_p"], targets["operand_bucket_p"])
        d_rs = tvd(m["result_bucket_p"], targets["result_bucket_p"])
        print("{:>4} {:>9.4f} {:>13.4f} {:>12.4f} {:>8}".format(it, d_op, d_bk, d_rs, m["n_equations"]))
        if d_op < 0.010 and d_bk < 0.012 and d_rs < 0.030:
            break
        for k in OPS:
            r = max(m["operator_p"].get(k, 1e-6), 1e-6)
            op_w[k] = max(1e-4, op_w[k] * (targets["operator_p"][k] / r) ** 0.7)
        for k in BUCKETS:
            r = max(m["operand_bucket_p"].get(k, 1e-6), 1e-6)
            bucket_w[k] = max(1e-4, bucket_w[k] * (targets["operand_bucket_p"][k] / r) ** 0.7)
            rr = max(m["result_bucket_p"].get(k, 1e-6), 1e-6)
            result_w[k] = min(1.0, max(0.02, result_w[k] * (targets["result_bucket_p"][k] / rr) ** 0.5))
        s = sum(op_w.values())
        op_w = {k: v / s for k, v in op_w.items()}
        s = sum(bucket_w.values())
        bucket_w = {k: v / s for k, v in bucket_w.items()}
        # keep the acceptance filter as permissive as possible
        mx = max(result_w.values())
        result_w = {k: v / mx for k, v in result_w.items()}
    return op_w, bucket_w, result_w


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            c = f.read(1 << 20)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


def write_split(traces: list, path: str, split: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, t in enumerate(traces):
            body = "\n".join(t["steps"])
            prompt = "Question: Calculate: " + t["expr"] + "\n"
            completion = "<think>\n" + body + "\n</think>\n<answer>" + t["answer"] + "</answer>"
            rec = {
                "id": "replay_{}_{:06d}".format(split, i),
                "dataset": "replay_arithmetic",
                "split": split,
                "condition": "operator_matched_replay",
                "question": "Calculate: " + t["expr"],
                "reasoning": body,
                "answer": t["answer"],
                "prompt": prompt,
                "completion": completion,
                "formatted_text": prompt + completion,
                "metadata": {"num_steps": len(t["steps"]), "steps": t["steps"], "verified": True},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def print_compare(title: str, realized: dict, target: dict, keys) -> None:
    print("")
    print(title)
    print("  {:<10} {:>9} {:>9} {:>8}".format("bucket", "replay", "gsm8k", "delta"))
    for k in keys:
        r = realized.get(k, 0.0) * 100
        t = target.get(k, 0.0) * 100
        print("  {:<10} {:>8.2f}% {:>8.2f}% {:>+7.2f}".format(str(k), r, t, r - t))
    print("  TVD = {:.4f}".format(tvd(realized, target)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 22 replay arithmetic corpus builder")
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_dev", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ipf_iters", type=int, default=10)
    ap.add_argument("--ipf_probe", type=int, default=2500)
    ap.add_argument("--source", default="data/gsm8k_compact_reasoning/train.jsonl")
    ap.add_argument("--out_dir", default="data/replay_arithmetic")
    args = ap.parse_args()

    print("=" * 72)
    print("MINI-R1 -- TASK 22: OPERATOR-MATCHED REPLAY ARITHMETIC CORPUS")
    print("=" * 72)

    targets = measure_targets(args.source)
    print("")
    print("Target measured from " + args.source)
    print("  traces={}  equations={}".format(targets["n_traces"], targets["n_equations"]))
    print("  operator_p : " + "  ".join("{}={:.2f}%".format(k, v * 100) for k, v in targets["operator_p"].items()))
    print("  operand_p  : " + "  ".join("{}={:.2f}%".format(k, v * 100) for k, v in targets["operand_bucket_p"].items()))
    print("  carry_rate : {:.2f}%   carry_left={:.1f}%".format(targets["carry_rate"] * 100, targets["carry_left_frac"] * 100))

    op_w, bucket_w, result_w = fit_weights(targets, args.seed, args.ipf_iters, args.ipf_probe)
    print("")
    print("Fitted fresh-sampling weights")
    print("  operators : " + "  ".join("{}={:.4f}".format(k, v) for k, v in op_w.items()))
    print("  buckets   : " + "  ".join("{}={:.4f}".format(k, v) for k, v in bucket_w.items()))
    print("  result_acc: " + "  ".join("{}={:.4f}".format(k, v) for k, v in result_w.items()))

    rng = random.Random(args.seed)
    total = args.n_train + args.n_dev
    print("")
    print("Generating {} verified traces...".format(total))
    all_traces = generate_pool(rng, total, targets, op_w, bucket_w, result_w)
    print("Generated + verified: {}".format(len(all_traces)))

    # Independent full re-verification pass.
    reasons: Counter = Counter()
    verified = []
    for t in all_traces:
        ok, why = verify_trace(t)
        if ok:
            verified.append(t)
        else:
            reasons[why] += 1
    print("Second-pass verification: {}/{} valid; failures={}".format(len(verified), len(all_traces), dict(reasons)))

    rng.shuffle(verified)
    dev = verified[: args.n_dev]
    train = verified[args.n_dev : args.n_dev + args.n_train]

    train_path = os.path.join(args.out_dir, "train.jsonl")
    dev_path = os.path.join(args.out_dir, "dev.jsonl")
    write_split(train, train_path, "train")
    write_split(dev, dev_path, "dev")

    m_train = measure_corpus(train)
    print("")
    print("=" * 72)
    print("REALIZED vs TARGET DISTRIBUTION (train split)")
    print("=" * 72)
    print_compare("OPERATOR FREQUENCY", m_train["operator_p"], targets["operator_p"], OPS)
    print_compare("OPERAND MAGNITUDE", m_train["operand_bucket_p"], targets["operand_bucket_p"], BUCKETS)
    print_compare("RESULT MAGNITUDE", m_train["result_bucket_p"], targets["result_bucket_p"], BUCKETS)
    print_compare("OPS PER EQUATION", m_train["arity_p"], targets["arity_p"], [1, 2, 3])
    print_compare(
        "EQUATIONS PER TRACE",
        m_train["steps_per_trace_p"],
        targets["steps_per_trace_p"],
        sorted(set(m_train["steps_per_trace_p"]) | set(targets["steps_per_trace_p"])),
    )

    manifest = {
        "dataset_name": "replay_arithmetic",
        "purpose": "Task 22 operator-matched arithmetic replay for parameter-isolated SFT",
        "source_distribution": args.source,
        "seed": args.seed,
        "n_train": len(train),
        "n_dev": len(dev),
        "n_train_equations": m_train["n_equations"],
        "verification": "100%_DETERMINISTICALLY_VERIFIED (per-equation + prompt-expression)",
        "second_pass_failures": dict(reasons),
        "fitted_weights": {"operator": op_w, "operand_bucket": bucket_w, "result_acceptance": result_w},
        "target_distribution": {k: v for k, v in targets.items() if k != "operand_pool"},
        "realized_distribution": m_train,
        "distribution_tvd": {
            "operator": tvd(m_train["operator_p"], targets["operator_p"]),
            "operand_bucket": tvd(m_train["operand_bucket_p"], targets["operand_bucket_p"]),
            "result_bucket": tvd(m_train["result_bucket_p"], targets["result_bucket_p"]),
            "arity": tvd(m_train["arity_p"], targets["arity_p"]),
            "steps_per_trace": tvd(m_train["steps_per_trace_p"], targets["steps_per_trace_p"]),
        },
        "files": {
            "train": {"path": train_path, "sha256": sha256_of(train_path)},
            "dev": {"path": dev_path, "sha256": sha256_of(dev_path)},
        },
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("")
    print("Wrote {} ({} examples)".format(train_path, len(train)))
    print("Wrote {} ({} examples)".format(dev_path, len(dev)))
    print("Wrote {}".format(os.path.join(args.out_dir, "manifest.json")))
    print("")
    print("Sample records:")
    for t in train[:4]:
        print("  PROMPT: Question: Calculate: " + t["expr"])
        for s in t["steps"]:
            print("          " + s)
        print("  ANSWER: " + t["answer"])
    print("=" * 72)


if __name__ == "__main__":
    main()
