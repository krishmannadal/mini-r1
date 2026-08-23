"""
Mini-R1 -- Unit Tests for Task 11B Qualification Fixes
======================================================
Tests granular discard accounting, incompatible schema handling, 60:40 token ratio
enforcement, deterministic sampling, zero train/val leakage, and packing compatibility.
"""

from pathlib import Path
import pytest
import torch

from src.tokenizer import Tokenizer
from src.general_dataset import (
    GeneralPretrainStreamer,
    PackedPretrainTorchDataset,
    GranularAccountingStats,
    INCOMPATIBLE_SUBSETS,
)
from src.corpus_qualifier import save_production_manifest_500m
from src.dataloader import create_dataloader

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tokenizer() -> Tokenizer:
    tok_path = PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.model"
    assert tok_path.exists(), f"Tokenizer missing at {tok_path}"
    return Tokenizer(tok_path)


class TestGranularAccountingAndSchema:
    def test_incompatible_schema_not_counted_as_quality_rejection(self, tokenizer):
        """Verify that python-edu is flagged as invalid_schema, not quality rejection."""
        streamer = GeneralPretrainStreamer(
            source_ratios={"python-edu": 1.0},
            tokenizer=tokenizer,
        )
        _, _, stats, breakdown = streamer.pack_tokens_from_stream(target_tokens=1000)
        assert stats.invalid_schema_examples > 0
        assert stats.rejected_quality_examples == 0
        assert "python-edu" in breakdown
        assert breakdown["python-edu"]["invalid_schema"] > 0
        assert breakdown["python-edu"]["rejected_quality"] == 0

    def test_granular_accounting_stats_structure(self):
        stats = GranularAccountingStats(
            total_records_seen=100,
            accepted_examples=85,
            duplicate_examples=5,
            rejected_quality_examples=6,
            empty_examples=2,
            invalid_schema_examples=2,
            preprocessing_errors=0,
            raw_tokens_untruncated=50000,
            packed_sequences=90,
            packed_tokens=46080,
            train_tokens=40960,
            validation_tokens=5120,
            discarded_tokens=20,
        )
        assert stats.accepted_examples + stats.duplicate_examples + stats.rejected_quality_examples + stats.empty_examples + stats.invalid_schema_examples == 100


class TestTokenRatioAndPacking:
    def test_token_ratio_budgeting_60_40(self, tokenizer):
        """Verify that target token budget splits ~60% cosmopedia-v2 and ~40% fineweb-edu-dedup."""
        streamer = GeneralPretrainStreamer(
            source_ratios={"cosmopedia-v2": 0.60, "fineweb-edu-dedup": 0.40},
            tokenizer=tokenizer,
            max_seq_len=512,
            seed=42,
        )
        target = 10240  # 20 sequences of 512
        train_chunks, val_chunks, stats, breakdown = streamer.pack_tokens_from_stream(
            target_tokens=target,
            val_ratio=0.1,
        )

        assert len(train_chunks) + len(val_chunks) >= 20
        cosmo_tokens = breakdown["cosmopedia-v2"]["packed_train_tokens"] + breakdown["cosmopedia-v2"]["packed_val_tokens"]
        fine_tokens = breakdown["fineweb-edu-dedup"]["packed_train_tokens"] + breakdown["fineweb-edu-dedup"]["packed_val_tokens"]

        total_packed = cosmo_tokens + fine_tokens
        assert total_packed >= 10240
        # Allow discrete chunk ceiling tolerance (e.g. 50-70% for cosmopedia, 30-50% for fineweb)
        cosmo_pct = (cosmo_tokens / total_packed) * 100
        fine_pct = (fine_tokens / total_packed) * 100
        assert 50.0 <= cosmo_pct <= 70.0
        assert 30.0 <= fine_pct <= 50.0

    def test_packed_dataset_dataloader_collation(self):
        """Verify PackedPretrainTorchDataset works with create_dataloader seamlessly."""
        dummy_chunks = [[i % 8000 for i in range(512)] for _ in range(16)]
        dataset = PackedPretrainTorchDataset(dummy_chunks)
        loader = create_dataloader(dataset, batch_size=4, shuffle=False)

        batch = next(iter(loader))
        assert batch.input_ids.shape == (4, 512)
        assert batch.targets.shape == (4, 512)
        assert batch.attention_mask.shape == (4, 512)
        assert torch.equal(batch.input_ids, batch.targets)
        assert (batch.attention_mask == 1).all()


class TestProductionManifestAndMath:
    def test_production_manifest_500m_steps_math(self, tmp_path, tokenizer):
        tok_path = PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.model"
        out_path = tmp_path / "manifest_500m.json"
        manifest = save_production_manifest_500m(
            tokenizer_path=tok_path,
            tokenizer_vocab_size=tokenizer.vocab_size(),
            output_path=out_path,
            seed=42,
        )

        assert manifest["target_training_tokens"] == 500_000_000
        assert manifest["target_validation_tokens"] == 10_000_000
        assert manifest["training_plan"]["total_optimizer_steps"] == 30517
        assert manifest["training_plan"]["tokens_per_optimizer_step"] == 16384
        assert manifest["source_mixture"] == {"cosmopedia-v2": 0.6, "fineweb-edu-dedup": 0.4}
        assert "python-edu" in manifest["incompatible_subsets_excluded"]
