"""
test_numeric_forensics.py - Comprehensive Unit Tests for Task 19A Numeric Representation Forensics.
"""

import json
import os
import pytest

from scripts.run_numeric_representation_forensics import (
    analyze_10k_integers,
    analyze_digit_ranges,
    analyze_train_vs_ood_tokenization,
    analyze_training_templates,
    classify_digit_representation,
    inspect_number_tokenization,
)
from src.tokenizer import Tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    tok_path = "artifacts/tokenizer/tokenizer.model"
    return Tokenizer(tok_path)


def test_digit_tokenization_integrity(tokenizer):
    """Verify single digits 0-9 are each 1 token, 2-digit are 2 tokens, 3-digit are 3 tokens."""
    for d in range(10):
        info = inspect_number_tokenization(tokenizer, str(d))
        assert info["num_tokens"] == 1
        assert len(info["pieces"]) == 1

    for num in [10, 25, 42, 99]:
        info = inspect_number_tokenization(tokenizer, str(num))
        assert info["num_tokens"] == 2

    for num in [100, 250, 641, 999]:
        info = inspect_number_tokenization(tokenizer, str(num))
        assert info["num_tokens"] == 3


def test_classify_digit_representation(tokenizer):
    """Verify digit representation classification."""
    rep = classify_digit_representation(tokenizer)
    assert rep["representation_type"] == "Pure Character / Digit-by-digit"
    assert rep["single_digits_present"] is True
    assert rep["total_pure_numeric_tokens"] == 20


def test_analyze_digit_ranges_and_10k_stats(tokenizer):
    """Verify range analysis and statistical aggregation."""
    ranges = analyze_digit_ranges(tokenizer)
    assert len(ranges["single_digits"]) == 10
    assert 2 in ranges["two_digit_token_length_dist"]

    stats = analyze_10k_integers(tokenizer, seed=42, n_samples=10000)
    assert stats["total_sampled"] == 10000
    assert stats["mean_tokens"] > 0
    assert "1_digits" in stats["by_digit_length"]
    assert "6_digits" in stats["by_digit_length"]


def test_train_vs_ood_token_overlap(tokenizer):
    """Verify train vs OOD token analysis metrics."""
    res = analyze_train_vs_ood_tokenization(tokenizer)
    assert res["train_numeric_token_ids_count"] == 10
    assert res["all_ood_numeric_token_ids_count"] == 10
    assert res["token_overlap_percentage"] == 100.0
    assert res["novel_token_ids_in_ood_count"] == 0


def test_training_template_analysis():
    """Verify training template analysis on compact_train.jsonl."""
    res = analyze_training_templates("data/arithmetic_sft/compact_train.jsonl")
    assert res["total_training_examples"] == 12000
    assert res["unique_expressions_count"] > 5000
    assert res["unique_operand_step_tuples_count"] > 5000


def test_forensic_artifacts_exist():
    """Verify JSON and Markdown report files exist and are populated."""
    json_path = "reports/task19a_numeric_representation_forensics.json"
    md_path = "reports/task19a_numeric_representation_forensics.md"

    assert os.path.exists(json_path)
    assert os.path.exists(md_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "tokenizer" in data
    assert "train_vs_ood_token_analysis" in data
    assert "training_distribution_template_analysis" in data
