"""
generate_arithmetic_ood_eval.py - Generates OOD Evaluation Datasets for Task 18B.

Generates:
1. data/arithmetic_sft/eval_ood_numbers.json (1,200 examples with operands in [101, 250], [26, 60], [13, 30])
2. data/arithmetic_sft/eval_ood_composition.json (1,200 examples with novel operator compositions)
3. data/arithmetic_sft/eval_word_wrapped.json (1,200 word-wrapped arithmetic problems)
"""

import json
import os
import random
from typing import Any


def gen_ood_numbers_example(rng: random.Random, level: int) -> dict[str, Any]:
    """Generates examples using strictly non-overlapping OOD operand ranges."""
    if level == 1:
        op_type = rng.choice(["add", "sub", "mul", "div"])
        if op_type == "add":
            a = rng.randint(101, 250)
            b = rng.randint(101, 250)
            res = a + b
            op = "+"
            expr = f"{a} + {b}"
        elif op_type == "sub":
            a = rng.randint(150, 250)
            b = rng.randint(101, a)
            res = a - b
            op = "-"
            expr = f"{a} - {b}"
        elif op_type == "mul":
            a = rng.randint(26, 60)
            b = rng.randint(10, 30)
            res = a * b
            op = "*"
            expr = f"{a} * {b}"
        else:  # div
            div = rng.randint(13, 30)
            res = rng.randint(10, 40)
            a = res * div
            op = "/"
            expr = f"{a} / {div}"
        steps = [{"step": 1, "lhs": a, "op": op, "rhs": b if op != "/" else div, "result": res, "state": res}]
        return {
            "problem_text": f"Calculate: {expr}",
            "expression": expr,
            "operations": steps,
            "final_answer": str(res),
            "op_count": 1,
        }
    elif level == 2:
        # OOD two-ops
        a = rng.randint(101, 200)
        b = rng.randint(101, 200)
        s1 = a + b
        c = rng.randint(101, s1 - 10)
        s2 = s1 - c
        expr = f"({a} + {b}) - {c}"
        steps = [
            {"step": 1, "lhs": a, "op": "+", "rhs": b, "result": s1, "state": s1},
            {"step": 2, "lhs": s1, "op": "-", "rhs": c, "result": s2, "state": s2},
        ]
        return {
            "problem_text": f"Calculate: {expr}",
            "expression": expr,
            "operations": steps,
            "final_answer": str(s2),
            "op_count": 2,
        }
    elif level == 3:
        a = rng.randint(26, 50)
        b = rng.randint(10, 25)
        s1 = a * b
        c = rng.randint(101, 200)
        s2 = s1 + c
        d = rng.randint(50, s2 - 10)
        s3 = s2 - d
        expr = f"(({a} * {b}) + {c}) - {d}"
        steps = [
            {"step": 1, "lhs": a, "op": "*", "rhs": b, "result": s1, "state": s1},
            {"step": 2, "lhs": s1, "op": "+", "rhs": c, "result": s2, "state": s2},
            {"step": 3, "lhs": s2, "op": "-", "rhs": d, "result": s3, "state": s3},
        ]
        return {
            "problem_text": f"Calculate: {expr}",
            "expression": expr,
            "operations": steps,
            "final_answer": str(s3),
            "op_count": 3,
        }
    elif level == 4:
        a = rng.randint(26, 40)
        b = rng.randint(10, 20)
        s1 = a * b
        c = rng.randint(101, 180)
        s2 = s1 + c
        d = rng.randint(50, s2 - 10)
        s3 = s2 - d
        e = rng.randint(2, 5)
        s4 = s3 * e
        expr = f"((({a} * {b}) + {c}) - {d}) * {e}"
        steps = [
            {"step": 1, "lhs": a, "op": "*", "rhs": b, "result": s1, "state": s1},
            {"step": 2, "lhs": s1, "op": "+", "rhs": c, "result": s2, "state": s2},
            {"step": 3, "lhs": s2, "op": "-", "rhs": d, "result": s3, "state": s3},
            {"step": 4, "lhs": s3, "op": "*", "rhs": e, "result": s4, "state": s4},
        ]
        return {
            "problem_text": f"Calculate: {expr}",
            "expression": expr,
            "operations": steps,
            "final_answer": str(s4),
            "op_count": 4,
        }
    elif level == 5:
        start = rng.randint(101, 200)
        curr = start
        steps = []
        text_parts = [f"Start with {start}."]
        for i in range(3):
            op = rng.choice(["add", "sub"])
            if op == "add":
                val = rng.randint(20, 100)
                nxt = curr + val
                steps.append({"step": i + 1, "lhs": curr, "op": "+", "rhs": val, "result": nxt, "state": nxt})
                text_parts.append(f"Add {val}.")
            else:
                max_sub = max(5, min(curr - 10, 80))
                val = rng.randint(5, max_sub) if max_sub >= 5 else 5
                nxt = curr - val
                steps.append({"step": i + 1, "lhs": curr, "op": "-", "rhs": val, "result": nxt, "state": nxt})
                text_parts.append(f"Subtract {val}.")
            curr = nxt
        text_parts.append("What is the final value?")
        return {
            "problem_text": " ".join(text_parts),
            "expression": f"Sequential({start} -> {curr})",
            "operations": steps,
            "final_answer": str(curr),
            "op_count": 3,
        }
    else:  # level 6: word-wrapped OOD
        a = rng.randint(101, 200)
        b = rng.randint(101, 200)
        res = a + b
        text = f"Question: Liam has {a} apples and receives {b} more apples. How many apples does Liam have in total?"
        steps = [{"step": 1, "lhs": a, "op": "+", "rhs": b, "result": res, "state": res}]
        return {
            "problem_text": text,
            "expression": f"{a} + {b}",
            "operations": steps,
            "final_answer": str(res),
            "op_count": 1,
        }


def gen_ood_composition_example(rng: random.Random) -> dict[str, Any]:
    """Generates unseen operator compositions (e.g. (a + b) * (c - d), etc.)."""
    pattern = rng.choice(["two_paren_mul", "sub_then_div_then_add", "add_then_mul_then_div"])
    if pattern == "two_paren_mul":
        a = rng.randint(10, 40)
        b = rng.randint(5, 30)
        s1 = a + b
        c = rng.randint(15, 40)
        d = rng.randint(2, c - 2)
        s2 = c - d
        s3 = s1 * s2
        expr = f"({a} + {b}) * ({c} - {d})"
        steps = [
            {"step": 1, "lhs": a, "op": "+", "rhs": b, "result": s1, "state": s1},
            {"step": 2, "lhs": c, "op": "-", "rhs": d, "result": s2, "state": s2},
            {"step": 3, "lhs": s1, "op": "*", "rhs": s2, "result": s3, "state": s3},
        ]
        return {
            "problem_text": f"Calculate: {expr}",
            "expression": expr,
            "operations": steps,
            "final_answer": str(s3),
            "op_count": 3,
        }
    elif pattern == "sub_then_div_then_add":
        div = rng.randint(2, 8)
        q = rng.randint(4, 20)
        s1 = q * div
        b = rng.randint(5, 30)
        a = s1 + b
        c = rng.randint(10, 50)
        s2 = q
        s3 = s2 + c
        expr = f"(({a} - {b}) / {div}) + {c}"
        steps = [
            {"step": 1, "lhs": a, "op": "-", "rhs": b, "result": s1, "state": s1},
            {"step": 2, "lhs": s1, "op": "/", "rhs": div, "result": s2, "state": s2},
            {"step": 3, "lhs": s2, "op": "+", "rhs": c, "result": s3, "state": s3},
        ]
        return {
            "problem_text": f"Calculate: {expr}",
            "expression": expr,
            "operations": steps,
            "final_answer": str(s3),
            "op_count": 3,
        }
    else:  # add_then_mul_then_div
        div = rng.randint(2, 6)
        factor = rng.randint(2, 5)
        # We need (s1 * factor) % div == 0
        s1 = div * rng.randint(3, 15)
        b = rng.randint(2, s1 - 2)
        a = s1 - b
        s2 = s1 * factor
        s3 = s2 // div
        expr = f"(({a} + {b}) * {factor}) / {div}"
        steps = [
            {"step": 1, "lhs": a, "op": "+", "rhs": b, "result": s1, "state": s1},
            {"step": 2, "lhs": s1, "op": "*", "rhs": factor, "result": s2, "state": s2},
            {"step": 3, "lhs": s2, "op": "/", "rhs": div, "result": s3, "state": s3},
        ]
        return {
            "problem_text": f"Calculate: {expr}",
            "expression": expr,
            "operations": steps,
            "final_answer": str(s3),
            "op_count": 3,
        }


def gen_word_wrapped_example(rng: random.Random) -> dict[str, Any]:
    """Generates complex word problem wrapping."""
    a = rng.randint(10, 35)
    b = rng.randint(3, 12)
    s1 = a * b
    c = rng.randint(15, 60)
    s2 = s1 + c
    d = rng.randint(5, min(40, s2 - 5))
    s3 = s2 - d
    text = (
        f"Question: A bookstore owner ordered {a} boxes containing {b} mystery novels each. "
        f"She also unpacked {c} individual hardcover novels. If she sold {d} novels on opening day, "
        f"how many novels remain in the store?"
    )
    steps = [
        {"step": 1, "lhs": a, "op": "*", "rhs": b, "result": s1, "state": s1},
        {"step": 2, "lhs": s1, "op": "+", "rhs": c, "result": s2, "state": s2},
        {"step": 3, "lhs": s2, "op": "-", "rhs": d, "result": s3, "state": s3},
    ]
    return {
        "problem_text": text,
        "expression": f"(({a} * {b}) + {c}) - {d}",
        "operations": steps,
        "final_answer": str(s3),
        "op_count": 3,
    }


def main():
    out_dir = os.path.join("data", "arithmetic_sft")
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(2026)

    # 1. OOD Numbers (1,200 examples, 200 per level)
    ood_num_records = []
    for level in range(1, 7):
        for i in range(200):
            item = gen_ood_numbers_example(rng, level)
            prompt = f"Question: {item['problem_text']}\n" if not item['problem_text'].startswith("Question:") else f"{item['problem_text']}\n"
            ood_num_records.append({
                "example_id": f"ood_num_L{level}_{i+1:04d}",
                "level": level,
                "prompt": prompt,
                "expression": item["expression"],
                "final_answer": item["final_answer"],
                "op_count": item["op_count"],
                "operations": item["operations"],
            })

    with open(os.path.join(out_dir, "eval_ood_numbers.json"), "w", encoding="utf-8") as f:
        json.dump(ood_num_records, f, indent=2)
    print(f"Generated eval_ood_numbers.json ({len(ood_num_records)} items)")

    # 2. OOD Composition (1,200 examples)
    ood_comp_records = []
    for i in range(1200):
        item = gen_ood_composition_example(rng)
        prompt = f"Question: {item['problem_text']}\n" if not item['problem_text'].startswith("Question:") else f"{item['problem_text']}\n"
        ood_comp_records.append({
            "example_id": f"ood_comp_{i+1:04d}",
            "prompt": prompt,
            "expression": item["expression"],
            "final_answer": item["final_answer"],
            "op_count": item["op_count"],
            "operations": item["operations"],
        })

    with open(os.path.join(out_dir, "eval_ood_composition.json"), "w", encoding="utf-8") as f:
        json.dump(ood_comp_records, f, indent=2)
    print(f"Generated eval_ood_composition.json ({len(ood_comp_records)} items)")

    # 3. Word-Wrapped (1,200 examples)
    word_records = []
    for i in range(1200):
        item = gen_word_wrapped_example(rng)
        prompt = f"{item['problem_text']}\n"
        word_records.append({
            "example_id": f"eval_word_{i+1:04d}",
            "prompt": prompt,
            "expression": item["expression"],
            "final_answer": item["final_answer"],
            "op_count": item["op_count"],
            "operations": item["operations"],
        })

    with open(os.path.join(out_dir, "eval_word_wrapped.json"), "w", encoding="utf-8") as f:
        json.dump(word_records, f, indent=2)
    print(f"Generated eval_word_wrapped.json ({len(word_records)} items)")


if __name__ == "__main__":
    main()
