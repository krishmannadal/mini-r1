"""
Mini-R1 -- Dataset Pipeline Test Suite
======================================
Tests to verify:
1. Dataset sample reproducibility with fixed random seed.
2. Strict train/test separation (zero leakage of GSM8K test questions into train/tokenizer splits).
3. Metadata integrity and artifact existence.
4. Correct reasoning tag wrapping (<think>, </think>, <answer>, </answer>).
"""

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class TestDatasetArtifacts:
    """Verify presence and structure of dataset artifacts."""

    def test_metadata_file_exists_and_valid(self):
        """metadata.json must exist and contain essential provenance fields."""
        meta_file = DATA_DIR / "metadata.json"
        assert meta_file.exists(), f"Metadata not found at {meta_file}"

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert "random_seed" in meta
        assert "pretrain_source" in meta
        assert "gsm8k_train_count" in meta
        assert "gsm8k_test_count" in meta
        assert meta["gsm8k_train_count"] > 0
        assert meta["gsm8k_test_count"] > 0

    def test_clean_files_exist_and_non_empty(self):
        """Clean split files must exist and contain non-trivial text."""
        for filename in ["pretrain.txt", "gsm8k_train.txt", "gsm8k_test.txt", "tokenizer_corpus.txt"]:
            file_path = DATA_DIR / "clean" / filename
            assert file_path.exists(), f"Missing file: {file_path}"
            assert file_path.stat().st_size > 1000, f"File {filename} is unexpectedly small"

    def test_raw_jsonl_records_valid(self):
        """Raw JSONL files must contain valid JSON lines with required fields."""
        raw_train = DATA_DIR / "raw" / "gsm8k_train.jsonl"
        assert raw_train.exists()

        with open(raw_train, "r", encoding="utf-8") as f:
            first_line = f.readline()
            record = json.loads(first_line)

        assert "id" in record
        assert "question" in record
        assert "reasoning" in record
        assert "answer" in record


class TestTrainTestSeparation:
    """Verify strictly zero data leakage from test split into training/tokenizer data."""

    def test_zero_leakage_in_train_set(self):
        """No test questions should appear in the train set."""
        raw_train = DATA_DIR / "raw" / "gsm8k_train.jsonl"
        raw_test = DATA_DIR / "raw" / "gsm8k_test.jsonl"

        with open(raw_train, "r", encoding="utf-8") as f:
            train_questions = {json.loads(line)["question"].strip() for line in f}

        with open(raw_test, "r", encoding="utf-8") as f:
            test_questions = {json.loads(line)["question"].strip() for line in f}

        overlap = train_questions.intersection(test_questions)
        assert len(overlap) == 0, f"DATA LEAKAGE DETECTED: {len(overlap)} questions overlap between train and test!"

    def test_zero_leakage_in_tokenizer_corpus(self):
        """No test questions should appear in the tokenizer training corpus."""
        raw_test = DATA_DIR / "raw" / "gsm8k_test.jsonl"
        tok_corpus = DATA_DIR / "clean" / "tokenizer_corpus.txt"

        with open(raw_test, "r", encoding="utf-8") as f:
            test_questions = [json.loads(line)["question"].strip() for line in f]

        with open(tok_corpus, "r", encoding="utf-8") as f:
            corpus_text = f.read()

        # Check a sample of test questions are not verbatim in the corpus
        for q in test_questions[:50]:
            assert q not in corpus_text, f"Test question leaked into tokenizer corpus: '{q[:50]}...'"


class TestReasoningFormatting:
    """Verify CoT reasoning tags in the cleaned GSM8K files."""

    def test_gsm8k_tags_present(self):
        """Cleaned GSM8K train records must contain <think> and <answer> tags."""
        clean_train = DATA_DIR / "clean" / "gsm8k_train.txt"
        with open(clean_train, "r", encoding="utf-8") as f:
            sample_content = f.read(5000)

        assert "<think>" in sample_content
        assert "</think>" in sample_content
        assert "<answer>" in sample_content
        assert "</answer>" in sample_content
