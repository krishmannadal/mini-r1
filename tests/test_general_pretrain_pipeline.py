"""
Mini-R1 -- Unit Tests for General Pretraining Data Pipeline
===========================================================
Tests streaming, token packing, exact token accounting, train/val split isolation,
deterministic seeding, and manifest generation.
"""

from pathlib import Path
import pytest
import torch

from src.tokenizer import Tokenizer
from src.general_dataset import (
    GeneralPretrainStreamer,
    clean_general_text,
    create_pretrain_manifest,
    TokenAccountingStats,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tokenizer() -> Tokenizer:
    tok_path = PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.model"
    assert tok_path.exists(), f"Tokenizer missing at {tok_path}"
    return Tokenizer(tok_path)


class TestGeneralTextCleaning:
    def test_whitespace_normalization(self):
        raw = "  Hello \r\n\r\n\r\nWorld!  \n\n\n\nTest.  "
        cleaned = clean_general_text(raw)
        assert cleaned == "Hello \n\nWorld!  \n\nTest."

    def test_empty_string(self):
        assert clean_general_text("") == ""


class TestDeterministicSamplingAndPacking:
    def test_packing_chunk_length_exact(self, tokenizer):
        """Verify that every packed sequence has length exactly equal to max_seq_len."""
        streamer = GeneralPretrainStreamer(
            dataset_name="HuggingFaceTB/smollm-corpus",
            subsets=["cosmopedia-v2"],
            tokenizer=tokenizer,
            max_seq_len=512,
            seed=42,
        )

        # Pack small quota for testing (e.g. 5,000 tokens)
        train_chunks, val_chunks, stats = streamer.pack_tokens_from_stream(
            target_tokens=5120,
            val_ratio=0.2,
        )

        assert len(train_chunks) > 0
        assert len(val_chunks) > 0
        for chunk in train_chunks:
            assert len(chunk) == 512
        for chunk in val_chunks:
            assert len(chunk) == 512

        assert stats.packed_tokens == (len(train_chunks) + len(val_chunks)) * 512
        assert stats.train_tokens == len(train_chunks) * 512
        assert stats.validation_tokens == len(val_chunks) * 512

    def test_deterministic_stream_reproducibility(self, tokenizer):
        """Verify identical seed produces exact identical token streams."""
        streamer1 = GeneralPretrainStreamer(
            subsets=["cosmopedia-v2"],
            tokenizer=tokenizer,
            max_seq_len=512,
            seed=123,
        )
        streamer2 = GeneralPretrainStreamer(
            subsets=["cosmopedia-v2"],
            tokenizer=tokenizer,
            max_seq_len=512,
            seed=123,
        )

        train1, val1, _ = streamer1.pack_tokens_from_stream(target_tokens=2048, val_ratio=0.25)
        train2, val2, _ = streamer2.pack_tokens_from_stream(target_tokens=2048, val_ratio=0.25)

        assert train1 == train2
        assert val1 == val2

    def test_train_val_separation(self, tokenizer):
        """Verify train and val chunks are distinct and have zero identical 512-blocks."""
        streamer = GeneralPretrainStreamer(
            subsets=["cosmopedia-v2"],
            tokenizer=tokenizer,
            max_seq_len=512,
            seed=42,
        )
        train_chunks, val_chunks, _ = streamer.pack_tokens_from_stream(target_tokens=4096, val_ratio=0.25)
        train_set = {tuple(c) for c in train_chunks}
        val_set = {tuple(c) for c in val_chunks}
        assert train_set.isdisjoint(val_set)

    def test_no_reasoning_tags_leakage_into_raw_general_text(self, tokenizer):
        """Verify that general text from smollm does not contain artificial <think> tags."""
        streamer = GeneralPretrainStreamer(
            subsets=["cosmopedia-v2"],
            tokenizer=tokenizer,
            max_seq_len=512,
            seed=42,
        )
        train_chunks, _, _ = streamer.pack_tokens_from_stream(target_tokens=2048, val_ratio=0.1)
        think_id = tokenizer.think_start_id
        # Raw cosmopedia documents should not have special reasoning delimiters
        for chunk in train_chunks:
            assert think_id not in chunk


class TestManifestGeneration:
    def test_manifest_structure_and_types(self, tmp_path, tokenizer):
        stats = TokenAccountingStats(
            raw_examples=100,
            cleaned_examples=98,
            raw_tokens_untruncated=50000,
            packed_sequences=90,
            packed_tokens=46080,
            train_tokens=40960,
            validation_tokens=5120,
            discarded_tokens=20,
            token_utilization_ratio=0.92,
        )

        manifest_file = tmp_path / "manifest.json"
        tok_path = PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.model"
        manifest = create_pretrain_manifest(
            target_token_budget=50000,
            stats=stats,
            tokenizer_path=tok_path,
            tokenizer_vocab_size=8000,
            subsets=["cosmopedia-v2"],
            seed=42,
            output_path=manifest_file,
        )

        assert manifest_file.exists()
        assert manifest["source_dataset"] == "HuggingFaceTB/smollm-corpus"
        assert manifest["token_statistics"]["packed_tokens"] == 46080
        assert manifest["splits"]["train_tokens"] == 40960
        assert manifest["splits"]["validation_tokens"] == 5120
        assert manifest["tokenizer"]["vocab_size"] == 8000
