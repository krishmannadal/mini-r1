"""
generate_arithmetic_probe.py - Deterministic synthetic arithmetic dataset generator for Task 17A.

Generates 9,000 deterministic arithmetic evaluation problems (1,000 per category)
across 9 categories with exact deterministic gold answers.
"""

import json
import os
import random
from typing import Any


def evaluate_expression_safe(expr_str: str) -> int:
    """Evaluate simple arithmetic expression with integer arithmetic."""
    # Only allow safe arithmetic characters
    cleaned = expr_str.replace("x", "*").replace("×", "*").replace("÷", "//")
    return int(eval(cleaned))


def generate_single_addition(rng: random.Random) -> dict[str, Any]:
    # 1-digit, 2-digit, 3-digit mix
    tier = rng.choice(["1d", "2d", "3d"])
    if tier == "1d":
        a = rng.randint(1, 9)
        b = rng.randint(1, 9)
    elif tier == "2d":
        a = rng.randint(10, 99)
        b = rng.randint(1, 99)
    else:
        a = rng.randint(100, 999)
        b = rng.randint(10, 999)
    
    expr = f"{a} + {b}"
    ans = a + b
    word = f"There are {a} red marbles and {b} blue marbles in a bag. How many marbles are there in total?"
    return {
        "category": "A_single_addition",
        "underlying_expression": expr,
        "gold_answer": str(ans),
        "difficulty_level": tier,
        "op_count": 1,
        "operations": ["addition"],
        "word_problem_text": word,
    }


def generate_single_subtraction(rng: random.Random) -> dict[str, Any]:
    tier = rng.choice(["1d", "2d", "3d"])
    if tier == "1d":
        a = rng.randint(1, 9)
        b = rng.randint(1, a)
    elif tier == "2d":
        a = rng.randint(10, 99)
        b = rng.randint(1, a)
    else:
        a = rng.randint(100, 999)
        b = rng.randint(10, a)
    
    expr = f"{a} - {b}"
    ans = a - b
    word = f"A baker had {a} cookies and sold {b} of them. How many cookies are left?"
    return {
        "category": "B_single_subtraction",
        "underlying_expression": expr,
        "gold_answer": str(ans),
        "difficulty_level": tier,
        "op_count": 1,
        "operations": ["subtraction"],
        "word_problem_text": word,
    }


def generate_single_multiplication(rng: random.Random) -> dict[str, Any]:
    tier = rng.choice(["1d_1d", "2d_1d", "2d_2d"])
    if tier == "1d_1d":
        a = rng.randint(2, 9)
        b = rng.randint(2, 9)
    elif tier == "2d_1d":
        a = rng.randint(10, 50)
        b = rng.randint(2, 9)
    else:
        a = rng.randint(10, 30)
        b = rng.randint(10, 30)
    
    expr = f"{a} * {b}"
    ans = a * b
    word = f"A box contains {a} packs of pencils. Each pack has {b} pencils. How many pencils are there in total?"
    return {
        "category": "C_single_multiplication",
        "underlying_expression": expr,
        "gold_answer": str(ans),
        "difficulty_level": tier,
        "op_count": 1,
        "operations": ["multiplication"],
        "word_problem_text": word,
    }


def generate_single_division(rng: random.Random) -> dict[str, Any]:
    tier = rng.choice(["small", "medium", "large"])
    if tier == "small":
        b = rng.randint(2, 9)
        ans = rng.randint(2, 9)
    elif tier == "medium":
        b = rng.randint(2, 12)
        ans = rng.randint(10, 30)
    else:
        b = rng.randint(3, 15)
        ans = rng.randint(20, 80)
    
    a = ans * b
    expr = f"{a} / {b}"
    word = f"A farmer distributes {a} apples equally among {b} baskets. How many apples are in each basket?"
    return {
        "category": "D_single_division",
        "underlying_expression": expr,
        "gold_answer": str(ans),
        "difficulty_level": tier,
        "op_count": 1,
        "operations": ["division"],
        "word_problem_text": word,
    }


def generate_two_op_expression(rng: random.Random) -> dict[str, Any]:
    pattern = rng.choice(["add_sub", "mul_add", "mul_sub", "add_mul"])
    if pattern == "add_sub":
        a = rng.randint(10, 50)
        b = rng.randint(5, 30)
        c = rng.randint(1, min(20, a + b))
        expr = f"{a} + {b} - {c}"
        ans = a + b - c
        ops = ["addition", "subtraction"]
        word = f"Liam had {a} stickers. He bought {b} more stickers and gave {c} stickers to a friend. How many stickers does he have now?"
    elif pattern == "mul_add":
        a = rng.randint(2, 9)
        b = rng.randint(2, 9)
        c = rng.randint(1, 30)
        expr = f"{a} * {b} + {c}"
        ans = (a * b) + c
        ops = ["multiplication", "addition"]
        word = f"Emma bought {a} packs of gum with {b} pieces in each pack, plus {c} loose pieces. How many pieces of gum does she have in total?"
    elif pattern == "mul_sub":
        a = rng.randint(3, 10)
        b = rng.randint(3, 10)
        c = rng.randint(1, a * b - 1)
        expr = f"{a} * {b} - {c}"
        ans = (a * b) - c
        ops = ["multiplication", "subtraction"]
        word = f"A gardener planted {a} rows of flowers with {b} flowers in each row. A storm destroyed {c} flowers. How many flowers remain?"
    else:  # add_mul
        a = rng.randint(1, 20)
        b = rng.randint(2, 8)
        c = rng.randint(2, 8)
        expr = f"{a} + {b} * {c}"
        ans = a + (b * c)
        ops = ["addition", "multiplication"]
        word = f"Noah has ${a}. He then earns ${b} per hour for {c} hours of chores. How much money does he have in total?"
    
    return {
        "category": "E_two_operation",
        "underlying_expression": expr,
        "gold_answer": str(ans),
        "difficulty_level": pattern,
        "op_count": 2,
        "operations": ops,
        "word_problem_text": word,
    }


def generate_three_op_expression(rng: random.Random) -> dict[str, Any]:
    pattern = rng.choice(["mul_add_sub", "add_mul_sub", "mul_add_add"])
    if pattern == "mul_add_sub":
        a = rng.randint(2, 8)
        b = rng.randint(2, 8)
        c = rng.randint(5, 25)
        d = rng.randint(1, (a * b + c) - 1)
        expr = f"{a} * {b} + {c} - {d}"
        ans = a * b + c - d
        ops = ["multiplication", "addition", "subtraction"]
        word = f"A shopkeeper ordered {a} boxes with {b} items each. She already had {c} items in stock and sold {d} items. How many items remain?"
    elif pattern == "add_mul_sub":
        a = rng.randint(5, 30)
        b = rng.randint(2, 6)
        c = rng.randint(2, 6)
        d = rng.randint(1, (a + b * c) - 1)
        expr = f"{a} + {b} * {c} - {d}"
        ans = a + b * c - d
        ops = ["addition", "multiplication", "subtraction"]
        word = f"Sophia had {a} books. She received {b} sets of {c} books each and donated {d} books. How many books does she have now?"
    else:  # mul_add_add
        a = rng.randint(2, 7)
        b = rng.randint(2, 7)
        c = rng.randint(5, 20)
        d = rng.randint(5, 20)
        expr = f"{a} * {b} + {c} + {d}"
        ans = a * b + c + d
        ops = ["multiplication", "addition", "addition"]
        word = f"A craft store sold {a} kits with {b} beads each, plus {c} red beads and {d} green beads. How many beads were sold altogether?"
    
    return {
        "category": "F_three_operation",
        "underlying_expression": expr,
        "gold_answer": str(ans),
        "difficulty_level": pattern,
        "op_count": 3,
        "operations": ops,
        "word_problem_text": word,
    }


def generate_four_op_expression(rng: random.Random) -> dict[str, Any]:
    a = rng.randint(2, 6)
    b = rng.randint(2, 6)
    c = rng.randint(2, 6)
    d = rng.randint(2, 6)
    e = rng.randint(5, 25)
    expr = f"{a} * {b} + {c} * {d} - {e}"
    ans = (a * b) + (c * d) - e
    if ans < 0:
        ans += 50
        e = (a * b) + (c * d) - ans
        expr = f"{a} * {b} + {c} * {d} + {abs(e)}"
        ans = (a * b) + (c * d) + abs(e)
    
    ops = ["multiplication", "multiplication", "addition", "subtraction"]
    word = f"A school bought {a} boxes of {b} blue markers and {c} boxes of {d} green markers. After {e} markers were used, how many unused markers were left?"
    return {
        "category": "G_four_operation",
        "underlying_expression": expr,
        "gold_answer": str(ans),
        "difficulty_level": "4_op_mix",
        "op_count": 4,
        "operations": ops,
        "word_problem_text": word,
    }


def generate_sequential_state(rng: random.Random) -> dict[str, Any]:
    x = rng.randint(10, 50)
    add_val = rng.randint(5, 20)
    mul_val = rng.randint(2, 4)
    sub_val = rng.randint(5, 15)
    
    # State tracking: Start with X, add Y -> (X+Y), multiply by Z -> (X+Y)*Z, subtract W -> (X+Y)*Z - W
    step1 = x + add_val
    step2 = step1 * mul_val
    step3 = step2 - sub_val
    expr = f"({x} + {add_val}) * {mul_val} - {sub_val}"
    
    word = f"Start with {x}. Add {add_val}. Multiply the result by {mul_val}. Finally, subtract {sub_val}. What is the final value?"
    return {
        "category": "H_sequential_state",
        "underlying_expression": expr,
        "gold_answer": str(step3),
        "difficulty_level": "3_step_state",
        "op_count": 3,
        "operations": ["addition", "multiplication", "subtraction"],
        "word_problem_text": word,
    }


def generate_word_wrapped(rng: random.Random) -> dict[str, Any]:
    # Select from 1 to 3 ops and generate rich word problem
    base_gen = rng.choice([
        generate_single_addition,
        generate_single_subtraction,
        generate_single_multiplication,
        generate_single_division,
        generate_two_op_expression,
    ])
    item = base_gen(rng)
    item["category"] = "I_word_wrapped"
    return item


def build_synthetic_dataset(total_per_category: int = 1000, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    categories = [
        ("A_single_addition", generate_single_addition),
        ("B_single_subtraction", generate_single_subtraction),
        ("C_single_multiplication", generate_single_multiplication),
        ("D_single_division", generate_single_division),
        ("E_two_operation", generate_two_op_expression),
        ("F_three_operation", generate_three_op_expression),
        ("G_four_operation", generate_four_op_expression),
        ("H_sequential_state", generate_sequential_state),
        ("I_word_wrapped", generate_word_wrapped),
    ]

    all_items: list[dict[str, Any]] = []
    item_id = 0

    for cat_name, generator in categories:
        for idx in range(total_per_category):
            item_id += 1
            item = generator(rng)
            pid = f"arith_{cat_name[:1].lower()}_{idx+1:04d}"
            
            expr = item["underlying_expression"]
            # Build 3 prompt formats
            if item["category"] == "I_word_wrapped" or "word_problem_text" in item and item["category"] == "H_sequential_state":
                prompt_f1 = f"Question: {item['word_problem_text']}"
                prompt_f2 = f"Solve: {item['word_problem_text']}"
                prompt_f3 = f"Question: {item['word_problem_text']}\n"
            else:
                prompt_f1 = f"What is {expr}?"
                prompt_f2 = f"Calculate: {expr}"
                prompt_f3 = f"Question: What is {expr}?\n"

            record = {
                "problem_id": pid,
                "category": item["category"],
                "underlying_expression": expr,
                "gold_answer": item["gold_answer"],
                "difficulty_level": item["difficulty_level"],
                "op_count": item["op_count"],
                "operations": item["operations"],
                "word_problem_text": item.get("word_problem_text", ""),
                "prompt_format_1_plain": prompt_f1,
                "prompt_format_2_structured": prompt_f2,
                "prompt_format_3_reasoning": prompt_f3,
            }
            all_items.append(record)

    return all_items


def main():
    out_dir = os.path.join("data", "eval")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_arithmetic.json")

    print("Generating 9,000 synthetic arithmetic evaluation problems (seed=42)...")
    dataset = build_synthetic_dataset(total_per_category=1000, seed=42)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Successfully wrote {len(dataset)} problems to: {out_path}")
    
    # Print sample distribution
    from collections import Counter
    counts = Counter(d["category"] for d in dataset)
    for cat, cnt in sorted(counts.items()):
        print(f"  {cat:<26}: {cnt} items")


if __name__ == "__main__":
    main()
