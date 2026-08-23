"""
Tests for Mini-R1 Base Model Evaluation Module (src/evaluation.py).
"""

from __future__ import annotations

import torch
import pytest

from src.model import MiniR1Transformer, MiniR1Config
from src.tokenizer import Tokenizer
from src.evaluation import (
    AnswerExtractor,
    FormatValidator,
    generate_autoregressive,
    select_qualitative_samples,
    ExampleEvalRecord,
)


class TestAnswerExtractor:
    """Test deterministic rule-based answer extraction across benchmarks."""

    def test_clean_numerical_string(self):
        assert AnswerExtractor.clean_numerical_string("$1,250") == "1250"
        assert AnswerExtractor.clean_numerical_string("42.0") == "42"
        assert AnswerExtractor.clean_numerical_string("3.14159") == "3.14159"
        assert AnswerExtractor.clean_numerical_string("85%") == "85"
        assert AnswerExtractor.clean_numerical_string("  -12.00  ") == "-12"
        assert AnswerExtractor.clean_numerical_string("100.") == "100"

    def test_extract_gsm8k_xml_tag(self):
        text = "<think>Some reasoning steps</think>\n<answer>42</answer>"
        ans, method = AnswerExtractor.extract_gsm8k_answer(text)
        assert ans == "42"
        assert method == "xml_answer_tag"

    def test_extract_gsm8k_hash_delimiter(self):
        text = "Therefore, the total cost is 50. #### 50"
        ans, method = AnswerExtractor.extract_gsm8k_answer(text)
        assert ans == "50"
        assert method == "gsm8k_hash_delimiter"

    def test_extract_gsm8k_phrasing(self):
        text = "After calculating, the final answer is $125."
        ans, method = AnswerExtractor.extract_gsm8k_answer(text)
        assert ans == "125"
        assert method == "answer_phrase"

    def test_extract_gsm8k_fallback_last_number(self):
        text = "She started with 10 apples and bought 5 more, so now has 15 apples in total."
        ans, method = AnswerExtractor.extract_gsm8k_answer(text)
        assert ans == "15"
        assert method == "last_number_fallback"

    def test_extract_gsm8k_failed(self):
        text = "No numbers anywhere in this output text."
        ans, method = AnswerExtractor.extract_gsm8k_answer(text)
        assert ans is None
        assert method == "extraction_failed"

    def test_extract_aqua_xml_tag(self):
        text = "<think>Calculations here</think>\n<answer>C</answer>"
        ans, method = AnswerExtractor.extract_aqua_answer(text)
        assert ans == "C"
        assert method == "xml_answer_tag"

    def test_extract_aqua_phrasing(self):
        text = "Hence, the correct answer is option B."
        ans, method = AnswerExtractor.extract_aqua_answer(text)
        assert ans == "B"
        assert method == "answer_phrase"

    def test_extract_aqua_fallback_letter(self):
        text = "We evaluate the choices: (A) 10, (B) 20, (C) 30. The correct choice is D"
        ans, method = AnswerExtractor.extract_aqua_answer(text)
        assert ans == "D"
        assert method in ("answer_phrase", "option_keyword", "last_capital_letter_fallback")

    def test_compare_answers_numerical(self):
        assert AnswerExtractor.compare_answers("42", "42", "numerical")
        assert AnswerExtractor.compare_answers("42.0", "42", "numerical")
        assert AnswerExtractor.compare_answers("$1,000", "1000", "numerical")
        assert AnswerExtractor.compare_answers("0.5", "0.500", "numerical")
        assert not AnswerExtractor.compare_answers("43", "42", "numerical")
        assert not AnswerExtractor.compare_answers(None, "42", "numerical")

    def test_compare_answers_multiple_choice(self):
        assert AnswerExtractor.compare_answers("A", "A", "multiple_choice")
        assert AnswerExtractor.compare_answers("b", "B", "multiple_choice")
        assert not AnswerExtractor.compare_answers("C", "D", "multiple_choice")
        assert not AnswerExtractor.compare_answers(None, "A", "multiple_choice")


class TestFormatValidator:
    """Test format compliance validation."""

    def test_fully_formatted(self):
        text = "Question: What is 2+2?\n<think>\n2+2=4\n</think>\n<answer>4</answer>"
        res = FormatValidator.validate_formatting(text)
        assert res["is_fully_formatted"] is True
        assert res["malformed_reason"] == "none"
        assert res["extracted_reasoning"] == "2+2=4"

    def test_unclosed_think_tag(self):
        text = "Question: What is 2+2?\n<think>\n2+2=4"
        res = FormatValidator.validate_formatting(text)
        assert res["is_fully_formatted"] is False
        assert res["malformed_reason"] == "unclosed_think_tag"

    def test_no_tags(self):
        text = "The answer is clearly 4."
        res = FormatValidator.validate_formatting(text)
        assert res["is_fully_formatted"] is False
        assert res["malformed_reason"] == "no_tags_present"

    def test_invalid_tag_order(self):
        text = "<answer>4</answer>\n<think>2+2=4</think>"
        res = FormatValidator.validate_formatting(text)
        assert res["is_fully_formatted"] is False
        assert res["malformed_reason"] == "invalid_tag_order"


class TestAutoregressiveGeneration:
    """Test generation loop on a tiny model configuration."""

    def test_generation_execution(self, tmp_path):
        config = MiniR1Config(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=2,
            d_ff=64,
            max_seq_len=64,
        )
        model = MiniR1Transformer(config)

        # Mock tokenizer
        class MockTokenizer:
            eos_id = 2
            def decode(self, token_ids):
                return " ".join(str(t) for t in token_ids)

        tok = MockTokenizer()
        prompt = [1, 10, 20]

        # Greedy generation
        out_greedy = generate_autoregressive(
            model=model,
            prompt_token_ids=prompt,
            tokenizer=tok,
            max_new_tokens=10,
            temperature=0.0,
            device="cpu",
        )
        assert out_greedy.num_generated_tokens <= 10
        assert out_greedy.num_prompt_tokens == 3
        assert len(out_greedy.generated_tokens) == out_greedy.num_generated_tokens

        # Sampling generation
        out_samp = generate_autoregressive(
            model=model,
            prompt_token_ids=prompt,
            tokenizer=tok,
            max_new_tokens=10,
            temperature=0.8,
            top_p=0.9,
            device="cpu",
        )
        assert out_samp.num_generated_tokens <= 10


class TestQualitativeSampleSelection:
    """Test qualitative categorizer."""

    def test_sampling_buckets(self):
        dummy_records = [
            ExampleEvalRecord(
                dataset="gsm8k",
                id=f"rec_{i}",
                question="What is 2+2?",
                gold_answer="4",
                decoding_strategy="greedy",
                prompt_text="Prompt",
                generated_text=f"Reasoning text step {i} <answer>{4 if i % 2 == 0 else 5}</answer>",
                extracted_reasoning="Reasoning text",
                extracted_answer="4" if i % 2 == 0 else "5",
                extraction_method="xml_answer_tag",
                extraction_success=True,
                is_correct=(i % 2 == 0),
                is_fully_formatted=(i % 3 == 0),
                malformed_reason="none" if (i % 3 == 0) else "no_tags_present",
                prompt_tokens=10,
                generated_tokens=20,
                latency_sec=0.05,
                stopped_by_eos=True,
            )
            for i in range(50)
        ]

        samples = select_qualitative_samples(dummy_records, target_count=20, seed=42)
        assert "correct_and_coherent" in samples
        assert "wrong_and_plausible" in samples
        assert "formatting_failure" in samples
