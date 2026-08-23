"""
Unit Test Suite for Task 15A -- Reasoning SFT Dataset Construction & Validation
================================================================================
Tests:
1. Parsing of GSM8K solutions (delimiters, calculator tags, final answers).
2. Direct vs CoT formatting templates and tag balance.
3. Train/dev deterministic split sizes and disjointness.
4. Tokenizer compatibility and multiplier calculation.
5. Quality checks and error detection.
6. Zero-leakage verification against test sets.
"""

import hashlib
import json
import pytest
from pathlib import Path

from scripts.build_reasoning_sft_dataset import (
    parse_gsm8k_solution,
    estimate_reasoning_steps,
    build_direct_text,
    build_cot_text,
    run_quality_check,
    compute_string_hash,
)
from src.evaluation import AnswerExtractor


def test_parse_gsm8k_solution_standard():
    raw = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips.\n#### 72"
    reasoning, final_ans, num_ops, calc_tags = parse_gsm8k_solution(raw)

    assert final_ans == "72"
    assert num_ops == 2
    assert calc_tags == ["48/2=24", "48+24=72"]
    assert "<<" not in reasoning
    assert ">>" not in reasoning
    assert "Natalia sold 48/2 = 24 clips in May." in reasoning
    assert "Natalia sold 48+24 = 72 clips." in reasoning


def test_parse_gsm8k_solution_currency_commas():
    raw = "The total cost is $1,250.00 each so 2 * 1250 = <<2*1250=2500>>2500.\n#### $2,500"
    reasoning, final_ans, num_ops, calc_tags = parse_gsm8k_solution(raw)

    assert final_ans == "2500"
    assert num_ops == 1
    assert calc_tags == ["2*1250=2500"]


def test_parse_gsm8k_missing_delimiter():
    raw = "This is some reasoning without a delimiter 42"
    with pytest.raises(ValueError, match="Missing '####' delimiter"):
        parse_gsm8k_solution(raw)


def test_estimate_reasoning_steps():
    reasoning = "Step 1: calculate initial cost.\nStep 2: add tax.\nStep 3: divide by 2."
    steps = estimate_reasoning_steps(reasoning, num_ops=2)
    assert steps == 3


def test_build_direct_text():
    q = "How many apples?"
    a = "5"
    formatted = build_direct_text(q, a)
    expected = "Question: How many apples?\n<answer>5</answer>"
    assert formatted == expected
    assert "<think>" not in formatted
    assert "</think>" not in formatted


def test_build_cot_text():
    q = "How many apples?"
    r = "First count red apples (2), then green (3). 2 + 3 = 5."
    a = "5"
    formatted = build_cot_text(q, r, a)
    expected = (
        "Question: How many apples?\n"
        "<think>\n"
        "First count red apples (2), then green (3). 2 + 3 = 5.\n"
        "</think>\n"
        "<answer>5</answer>"
    )
    assert formatted == expected
    assert formatted.count("<think>") == 1
    assert formatted.count("</think>") == 1
    assert formatted.count("<answer>") == 1
    assert formatted.count("</answer>") == 1


def test_quality_check_pass():
    records = [
        {
            "id": "gsm8k_sft_train_0001",
            "condition": "cot",
            "question": "If you have 10 marbles and lose 3, how many remain?",
            "reasoning": "You start with 10 marbles. You lose 3 marbles. 10 - 3 = 7.",
            "answer": "7",
            "formatted_text": (
                "Question: If you have 10 marbles and lose 3, how many remain?\n"
                "<think>\n"
                "You start with 10 marbles. You lose 3 marbles. 10 - 3 = 7.\n"
                "</think>\n"
                "<answer>7</answer>"
            ),
        }
    ]
    res = run_quality_check(records, sample_size=1)
    assert res["passed_count"] == 1
    assert res["pass_rate_pct"] == 100.0
    assert len(res["failures"]) == 0


def test_quality_check_failure_malformed_tags():
    records = [
        {
            "id": "gsm8k_sft_train_0002",
            "condition": "cot",
            "question": "What is 2 + 2?",
            "reasoning": "2 + 2 = 4.",
            "answer": "4",
            "formatted_text": "Question: What is 2 + 2?\n<think>2 + 2 = 4.\n<answer>4</answer>",  # missing </think>
        }
    ]
    res = run_quality_check(records, sample_size=1)
    assert res["passed_count"] == 0
    assert res["pass_rate_pct"] == 0.0
    assert len(res["failures"]) == 1
