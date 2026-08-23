"""
generate_verified_arithmetic_sft.py - Deterministic Computational SFT Dataset Generator for Task 18A.

Generates a 6-level curriculum arithmetic dataset (12,000 train / 1,200 dev):
Level 1: Single operations (addition, subtraction, multiplication, division)
Level 2: Two operations
Level 3: Three operations
Level 4: Four operations
Level 5: Sequential state transitions
Level 6: Word-wrapped arithmetic versions of Levels 1-5

Exports paired compact and verbose reasoning formats with machine-verifiable step metadata.
"""

import json
import os
import random
from typing import Any


def op_symbol_to_word(op: str) -> tuple[str, str]:
    """Returns (operator_verb_phrase, result_verb)."""
    if op == "+":
        return "plus", "equals"
    elif op == "-":
        return "minus", "leaves"
    elif op == "*":
        return "multiplied by", "gives"
    elif op == "/":
        return "divided by", "equals"
    raise ValueError(f"Unknown operator: {op}")


def format_compact_trace(operations: list[dict[str, Any]], final_answer: str) -> str:
    """Format step-by-step compact arithmetic trace."""
    lines = ["<think>"]
    for step in operations:
        lines.append(f"{step['lhs']} {step['op']} {step['rhs']} = {step['result']}")
    lines.append("</think>")
    lines.append(f"<answer>{final_answer}</answer>")
    return "\n".join(lines)


def format_verbose_trace(operations: list[dict[str, Any]], final_answer: str) -> str:
    """Format verbose natural-language reasoning trace."""
    lines = ["<think>"]
    for i, step in enumerate(operations):
        op_word, res_verb = op_symbol_to_word(step["op"])
        if i == 0:
            prefix = "First, "
        elif i == len(operations) - 1:
            prefix = "Finally, "
        else:
            prefix = "Then, "
        lines.append(f"{prefix}{step['lhs']} {op_word} {step['rhs']} {res_verb} {step['result']}.")
    lines.append("</think>")
    lines.append(f"<answer>{final_answer}</answer>")
    return "\n".join(lines)


# =====================================================================
# LEVEL GENERATORS (In-Distribution Train/Dev Ranges)
# =====================================================================

def gen_level_1_single(rng: random.Random) -> dict[str, Any]:
    op_type = rng.choice(["add", "sub", "mul", "div"])
    if op_type == "add":
        a = rng.randint(2, 99)
        b = rng.randint(2, 99)
        res = a + b
        op = "+"
        expr = f"{a} + {b}"
    elif op_type == "sub":
        a = rng.randint(5, 99)
        b = rng.randint(2, a)
        res = a - b
        op = "-"
        expr = f"{a} - {b}"
    elif op_type == "mul":
        a = rng.randint(2, 25)
        b = rng.randint(2, 15)
        res = a * b
        op = "*"
        expr = f"{a} * {b}"
    else:  # div
        b = rng.randint(2, 12)
        res = rng.randint(2, 25)
        a = res * b
        op = "/"
        expr = f"{a} / {b}"

    steps = [{
        "step": 1,
        "lhs": a,
        "op": op,
        "rhs": b,
        "result": res,
        "state": res,
    }]
    return {
        "problem_text": f"Calculate: {expr}",
        "expression": expr,
        "operations": steps,
        "final_answer": str(res),
        "op_count": 1,
        "op_types": [op_type],
    }


def gen_level_2_two_ops(rng: random.Random) -> dict[str, Any]:
    pattern = rng.choice(["add_sub", "mul_add", "mul_sub", "div_add", "add_mul"])
    steps = []

    if pattern == "add_sub":
        a = rng.randint(10, 80)
        b = rng.randint(5, 50)
        s1 = a + b
        c = rng.randint(2, s1 - 1)
        s2 = s1 - c
        expr = f"({a} + {b}) - {c}"
        steps = [
            {"step": 1, "lhs": a, "op": "+", "rhs": b, "result": s1, "state": s1},
            {"step": 2, "lhs": s1, "op": "-", "rhs": c, "result": s2, "state": s2},
        ]
        op_types = ["add", "sub"]
    elif pattern == "mul_add":
        a = rng.randint(2, 15)
        b = rng.randint(2, 12)
        s1 = a * b
        c = rng.randint(3, 50)
        s2 = s1 + c
        expr = f"({a} * {b}) + {c}"
        steps = [
            {"step": 1, "lhs": a, "op": "*", "rhs": b, "result": s1, "state": s1},
            {"step": 2, "lhs": s1, "op": "+", "rhs": c, "result": s2, "state": s2},
        ]
        op_types = ["mul", "add"]
    elif pattern == "mul_sub":
        a = rng.randint(3, 15)
        b = rng.randint(3, 12)
        s1 = a * b
        c = rng.randint(2, s1 - 1)
        s2 = s1 - c
        expr = f"({a} * {b}) - {c}"
        steps = [
            {"step": 1, "lhs": a, "op": "*", "rhs": b, "result": s1, "state": s1},
            {"step": 2, "lhs": s1, "op": "-", "rhs": c, "result": s2, "state": s2},
        ]
        op_types = ["mul", "sub"]
    elif pattern == "div_add":
        q = rng.randint(2, 20)
        div = rng.randint(2, 10)
        a = q * div
        s1 = q
        c = rng.randint(5, 50)
        s2 = s1 + c
        expr = f"({a} / {div}) + {c}"
        steps = [
            {"step": 1, "lhs": a, "op": "/", "rhs": div, "result": s1, "state": s1},
            {"step": 2, "lhs": s1, "op": "+", "rhs": c, "result": s2, "state": s2},
        ]
        op_types = ["div", "add"]
    else:  # add_mul
        a = rng.randint(2, 30)
        b = rng.randint(2, 20)
        s1 = a + b
        c = rng.randint(2, 8)
        s2 = s1 * c
        expr = f"({a} + {b}) * {c}"
        steps = [
            {"step": 1, "lhs": a, "op": "+", "rhs": b, "result": s1, "state": s1},
            {"step": 2, "lhs": s1, "op": "*", "rhs": c, "result": s2, "state": s2},
        ]
        op_types = ["add", "mul"]

    return {
        "problem_text": f"Calculate: {expr}",
        "expression": expr,
        "operations": steps,
        "final_answer": str(steps[-1]["result"]),
        "op_count": 2,
        "op_types": op_types,
    }


def gen_level_3_three_ops(rng: random.Random) -> dict[str, Any]:
    pattern = rng.choice(["mul_add_sub", "add_mul_sub", "mul_add_div", "add_sub_mul"])
    steps = []

    if pattern == "mul_add_sub":
        a = rng.randint(2, 12)
        b = rng.randint(2, 10)
        s1 = a * b
        c = rng.randint(5, 40)
        s2 = s1 + c
        d = rng.randint(2, s2 - 1)
        s3 = s2 - d
        expr = f"(({a} * {b}) + {c}) - {d}"
        steps = [
            {"step": 1, "lhs": a, "op": "*", "rhs": b, "result": s1, "state": s1},
            {"step": 2, "lhs": s1, "op": "+", "rhs": c, "result": s2, "state": s2},
            {"step": 3, "lhs": s2, "op": "-", "rhs": d, "result": s3, "state": s3},
        ]
        op_types = ["mul", "add", "sub"]
    elif pattern == "add_mul_sub":
        a = rng.randint(2, 20)
        b = rng.randint(2, 15)
        s1 = a + b
        c = rng.randint(2, 6)
        s2 = s1 * c
        d = rng.randint(2, s2 - 1)
        s3 = s2 - d
        expr = f"(({a} + {b}) * {c}) - {d}"
        steps = [
            {"step": 1, "lhs": a, "op": "+", "rhs": b, "result": s1, "state": s1},
            {"step": 2, "lhs": s1, "op": "*", "rhs": c, "result": s2, "state": s2},
            {"step": 3, "lhs": s2, "op": "-", "rhs": d, "result": s3, "state": s3},
        ]
        op_types = ["add", "mul", "sub"]
    elif pattern == "mul_add_div":
        div = rng.randint(2, 8)
        q = rng.randint(5, 25)
        target = q * div  # must be s2
        c = rng.randint(2, min(target - 2, 30))
        rem = target - c
        # find a * b = rem
        # pick factor a
        factors = [i for i in range(2, 13) if rem % i == 0]
        if not factors:
            # fallback simple
            a, b = 4, 5
            c = (target - 20) if target >= 20 else 4
            target = a * b + c
            div = 2 if target % 2 == 0 else 3
            target = (target // div) * div
            c = target - (a * b)
        else:
            a = rng.choice(factors)
            b = rem // a
        s1 = a * b
        s2 = s1 + c
        s3 = s2 // div
        expr = f"(({a} * {b}) + {c}) / {div}"
        steps = [
            {"step": 1, "lhs": a, "op": "*", "rhs": b, "result": s1, "state": s1},
            {"step": 2, "lhs": s1, "op": "+", "rhs": c, "result": s2, "state": s2},
            {"step": 3, "lhs": s2, "op": "/", "rhs": div, "result": s3, "state": s3},
        ]
        op_types = ["mul", "add", "div"]
    else:  # add_sub_mul
        a = rng.randint(15, 60)
        b = rng.randint(5, 30)
        s1 = a + b
        c = rng.randint(2, min(25, s1 - 2))
        s2 = s1 - c
        d = rng.randint(2, 5)
        s3 = s2 * d
        expr = f"(({a} + {b}) - {c}) * {d}"
        steps = [
            {"step": 1, "lhs": a, "op": "+", "rhs": b, "result": s1, "state": s1},
            {"step": 2, "lhs": s1, "op": "-", "rhs": c, "result": s2, "state": s2},
            {"step": 3, "lhs": s2, "op": "*", "rhs": d, "result": s3, "state": s3},
        ]
        op_types = ["add", "sub", "mul"]

    return {
        "problem_text": f"Calculate: {expr}",
        "expression": expr,
        "operations": steps,
        "final_answer": str(steps[-1]["result"]),
        "op_count": 3,
        "op_types": op_types,
    }


def gen_level_4_four_ops(rng: random.Random) -> dict[str, Any]:
    a = rng.randint(2, 15)
    b = rng.randint(2, 10)
    s1 = a * b
    c = rng.randint(5, 30)
    s2 = s1 + c
    d = rng.randint(2, min(20, s2 - 2))
    s3 = s2 - d
    e = rng.randint(2, 4)
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
        "op_types": ["mul", "add", "sub", "mul"],
    }


def gen_level_5_sequential(rng: random.Random) -> dict[str, Any]:
    num_steps = rng.choice([2, 3, 4])
    start = rng.randint(10, 50)
    curr = start
    steps = []
    text_parts = [f"Start with {start}."]
    op_types = []

    for i in range(num_steps):
        # Choose operation that keeps numbers positive and reasonable
        avail_ops = ["add", "sub", "mul"]
        if curr > 100:
            avail_ops = ["sub", "add"]
        op_choice = rng.choice(avail_ops)
        
        if op_choice == "add":
            val = rng.randint(3, 25)
            nxt = curr + val
            steps.append({"step": i + 1, "lhs": curr, "op": "+", "rhs": val, "result": nxt, "state": nxt})
            text_parts.append(f"Add {val}.")
            op_types.append("add")
        elif op_choice == "sub":
            val = rng.randint(2, max(2, min(curr - 1, 20)))
            nxt = curr - val
            steps.append({"step": i + 1, "lhs": curr, "op": "-", "rhs": val, "result": nxt, "state": nxt})
            text_parts.append(f"Subtract {val}.")
            op_types.append("sub")
        else:  # mul
            val = rng.randint(2, 4)
            nxt = curr * val
            steps.append({"step": i + 1, "lhs": curr, "op": "*", "rhs": val, "result": nxt, "state": nxt})
            text_parts.append(f"Multiply by {val}.")
            op_types.append("mul")
        curr = nxt

    text_parts.append("What is the final value?")
    problem_text = " ".join(text_parts)

    return {
        "problem_text": problem_text,
        "expression": f"Sequential({start} -> {curr})",
        "operations": steps,
        "final_answer": str(curr),
        "op_count": len(steps),
        "op_types": op_types,
    }


def gen_level_6_word_wrapped(rng: random.Random) -> dict[str, Any]:
    # Select underlying problem from levels 1-3 and wrap in natural language
    base_level = rng.choice([1, 2, 3])
    if base_level == 1:
        base = gen_level_1_single(rng)
        step = base["operations"][0]
        a, op, b = step["lhs"], step["op"], step["rhs"]
        if op == "+":
            w_text = f"Liam has {a} apples and receives {b} more apples. How many apples does Liam have in total?"
        elif op == "-":
            w_text = f"A library has {a} books on a shelf and lends out {b} books. How many books remain on the shelf?"
        elif op == "*":
            w_text = f"A store sells {a} boxes of markers with {b} markers in each box. How many markers are there in total?"
        else:  # div
            w_text = f"A teacher distributes {a} stickers equally among {b} students. How many stickers does each student receive?"
    elif base_level == 2:
        base = gen_level_2_two_ops(rng)
        s1, s2 = base["operations"][0], base["operations"][1]
        a, b, c = s1["lhs"], s1["rhs"], s2["rhs"]
        if s1["op"] == "*" and s2["op"] == "+":
            w_text = f"Emma bought {a} packs of notebooks with {b} notebooks per pack, plus {c} single notebooks. How many notebooks does Emma have altogether?"
        elif s1["op"] == "*" and s2["op"] == "-":
            w_text = f"A baker made {a} trays of cookies with {b} cookies on each tray. If {c} cookies were eaten, how many cookies are left?"
        elif s1["op"] == "+" and s2["op"] == "*":
            w_text = f"Noah has {a} blue pens and {b} red pens in a case. He buys {c} identical pencil cases. How many pens does he have in total?"
        else:
            w_text = f"Sophie collected {a} shells and found {b} more. She then gave {c} shells to her brother. How many shells does she keep?"
    else:  # level 3
        base = gen_level_3_three_ops(rng)
        s1, s2, s3 = base["operations"][0], base["operations"][1], base["operations"][2]
        a, b, c, d = s1["lhs"], s1["rhs"], s2["rhs"], s3["rhs"]
        w_text = f"A warehouse received {a} pallets of {b} boxes each. The manager found {c} loose boxes in the back and shipped out {d} boxes. How many boxes remain in the warehouse?"

    base["problem_text"] = f"Question: {w_text}"
    return base


LEVEL_GENERATORS = {
    1: ("Level 1: Single Operations", gen_level_1_single),
    2: ("Level 2: Two Operations", gen_level_2_two_ops),
    3: ("Level 3: Three Operations", gen_level_3_three_ops),
    4: ("Level 4: Four Operations", gen_level_4_four_ops),
    5: ("Level 5: Sequential State", gen_level_5_sequential),
    6: ("Level 6: Word-Wrapped", gen_level_6_word_wrapped),
}


def build_curriculum_dataset(
    train_per_level: int = 2000,
    dev_per_level: int = 200,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build deterministic train and dev splits across all 6 levels."""
    rng = random.Random(seed)
    train_records: list[dict[str, Any]] = []
    dev_records: list[dict[str, Any]] = []

    global_id = 0

    for level, (lvl_desc, generator) in LEVEL_GENERATORS.items():
        # Generate Train Set
        for idx in range(train_per_level):
            global_id += 1
            item = generator(rng)
            pid = f"arith_L{level}_train_{idx+1:05d}"
            
            prompt = f"Question: {item['problem_text']}\n" if not item['problem_text'].startswith("Question:") else f"{item['problem_text']}\n"
            compact_completion = format_compact_trace(item["operations"], item["final_answer"])
            verbose_completion = format_verbose_trace(item["operations"], item["final_answer"])

            record = {
                "example_id": pid,
                "curriculum_level": level,
                "level_description": lvl_desc,
                "split": "train",
                "problem_text": item["problem_text"],
                "prompt": prompt,
                "underlying_expression": item["expression"],
                "underlying_operations": item["operations"],
                "op_count": item["op_count"],
                "op_types": item["op_types"],
                "final_answer": item["final_answer"],
                "compact_completion": compact_completion,
                "verbose_completion": verbose_completion,
            }
            train_records.append(record)

        # Generate Dev Set
        for idx in range(dev_per_level):
            global_id += 1
            item = generator(rng)
            pid = f"arith_L{level}_dev_{idx+1:05d}"
            
            prompt = f"Question: {item['problem_text']}\n" if not item['problem_text'].startswith("Question:") else f"{item['problem_text']}\n"
            compact_completion = format_compact_trace(item["operations"], item["final_answer"])
            verbose_completion = format_verbose_trace(item["operations"], item["final_answer"])

            record = {
                "example_id": pid,
                "curriculum_level": level,
                "level_description": lvl_desc,
                "split": "dev",
                "problem_text": item["problem_text"],
                "prompt": prompt,
                "underlying_expression": item["expression"],
                "underlying_operations": item["operations"],
                "op_count": item["op_count"],
                "op_types": item["op_types"],
                "final_answer": item["final_answer"],
                "compact_completion": compact_completion,
                "verbose_completion": verbose_completion,
            }
            dev_records.append(record)

    return train_records, dev_records


def main():
    out_dir = os.path.join("data", "arithmetic_sft")
    os.makedirs(out_dir, exist_ok=True)

    print("Generating Verified Arithmetic SFT Dataset (seed=42)...")
    train_records, dev_records = build_curriculum_dataset(
        train_per_level=2000,
        dev_per_level=200,
        seed=42,
    )

    print(f"Total Train Examples: {len(train_records)} (2,000 per level x 6 levels)")
    print(f"Total Dev Examples:   {len(dev_records)} (200 per level x 6 levels)")

    # 1. Export Compact JSONL
    compact_train_path = os.path.join(out_dir, "compact_train.jsonl")
    compact_dev_path = os.path.join(out_dir, "compact_dev.jsonl")
    
    with open(compact_train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            item = {
                "example_id": r["example_id"],
                "prompt": r["prompt"],
                "completion": r["compact_completion"],
                "metadata": {
                    "curriculum_level": r["curriculum_level"],
                    "op_count": r["op_count"],
                    "final_answer": r["final_answer"],
                    "underlying_operations": r["underlying_operations"],
                }
            }
            f.write(json.dumps(item) + "\n")

    with open(compact_dev_path, "w", encoding="utf-8") as f:
        for r in dev_records:
            item = {
                "example_id": r["example_id"],
                "prompt": r["prompt"],
                "completion": r["compact_completion"],
                "metadata": {
                    "curriculum_level": r["curriculum_level"],
                    "op_count": r["op_count"],
                    "final_answer": r["final_answer"],
                    "underlying_operations": r["underlying_operations"],
                }
            }
            f.write(json.dumps(item) + "\n")

    # 2. Export Verbose JSONL
    verbose_train_path = os.path.join(out_dir, "verbose_train.jsonl")
    verbose_dev_path = os.path.join(out_dir, "verbose_dev.jsonl")

    with open(verbose_train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            item = {
                "example_id": r["example_id"],
                "prompt": r["prompt"],
                "completion": r["verbose_completion"],
                "metadata": {
                    "curriculum_level": r["curriculum_level"],
                    "op_count": r["op_count"],
                    "final_answer": r["final_answer"],
                    "underlying_operations": r["underlying_operations"],
                }
            }
            f.write(json.dumps(item) + "\n")

    with open(verbose_dev_path, "w", encoding="utf-8") as f:
        for r in dev_records:
            item = {
                "example_id": r["example_id"],
                "prompt": r["prompt"],
                "completion": r["verbose_completion"],
                "metadata": {
                    "curriculum_level": r["curriculum_level"],
                    "op_count": r["op_count"],
                    "final_answer": r["final_answer"],
                    "underlying_operations": r["underlying_operations"],
                }
            }
            f.write(json.dumps(item) + "\n")

    # 3. Export Manifest
    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = {
        "dataset_name": "Mini-R1-Verified-Arithmetic-SFT",
        "task": "Task 18A: Verified Computational Training Corpus",
        "seed": 42,
        "total_train_examples": len(train_records),
        "total_dev_examples": len(dev_records),
        "curriculum_levels": {
            "Level 1": "Single Operations (+, -, *, /)",
            "Level 2": "Two-Operation Compositions",
            "Level 3": "Three-Operation Compositions",
            "Level 4": "Four-Operation Compositions",
            "Level 5": "Sequential State Transitions",
            "Level 6": "Word-Wrapped Natural Language Problems",
        },
        "operand_ranges": {
            "train_dev_in_distribution": {
                "addition_subtraction": [2, 99],
                "multiplication_factors": [2, 25],
                "division_divisors": [2, 12],
            },
            "holdout_evaluation_out_of_distribution": {
                "addition_subtraction": [101, 250],
                "multiplication_factors": [26, 60],
                "division_divisors": [13, 30],
            }
        },
        "files": {
            "compact_train": "data/arithmetic_sft/compact_train.jsonl",
            "compact_dev": "data/arithmetic_sft/compact_dev.jsonl",
            "verbose_train": "data/arithmetic_sft/verbose_train.jsonl",
            "verbose_dev": "data/arithmetic_sft/verbose_dev.jsonl",
        }
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Dataset generated successfully in {out_dir}/")
    print(f"  - compact_train.jsonl: {len(train_records)} rows")
    print(f"  - compact_dev.jsonl:   {len(dev_records)} rows")
    print(f"  - verbose_train.jsonl: {len(train_records)} rows")
    print(f"  - verbose_dev.jsonl:   {len(dev_records)} rows")
    print(f"  - manifest.json:       written")


if __name__ == "__main__":
    main()
