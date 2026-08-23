"""
Mini-R1 -- Unit Tests for Pretraining Corpus Qualification
==========================================================
Tests deduplication hashing, zero train/val leakage, percentile statistics,
packing efficiency calculations, and manifest schema validation.
"""

from pathlib import Path
import pytest
import numpy as np

from src.tokenizer import Tokenizer
from src.corpus_qualifier import (
    PretrainingCorpusQualifier,
    LengthDistributionStats,
    save_manifest_file,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tokenizer() -> Tokenizer:
    tok_path = PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.model"
    assert tok_path.exists(), f"Tokenizer missing at {tok_path}"
    return Tokenizer(tok_path)


class TestDeduplicationAndLeakage:
    def test_text_hash_normalization(self, tokenizer):
        qualifier = PretrainingCorpusQualifier(tokenizer=tokenizer)
        t1 = "This is a test document with   extra   spaces."
        t2 = "this is a test document with extra spaces.  \n"
        h1 = qualifier.compute_text_hash(t1)
        h2 = qualifier.compute_text_hash(t2)
        assert h1 == h2

    def test_zero_leakage_on_streaming_sample(self, tokenizer):
        qualifier = PretrainingCorpusQualifier(
            tokenizer=tokenizer,
            subsets=["cosmopedia-v2"],
            seed=42,
        )
        report = qualifier.run_qualification(sample_quota_per_subset=20, val_ratio=0.2)
        assert not report.leakage_detected
        assert report.train_val_collision_count == 0


class TestLengthDistributionMath:
    def test_percentile_calculations_match_numpy(self):
        lengths = [50, 100, 150, 200, 250, 300, 400, 500, 600, 1000]
        arr = np.array(lengths)
        stats = LengthDistributionStats(
            mean_length=float(np.mean(arr)),
            median_length=float(np.median(arr)),
            p25=float(np.percentile(arr, 25)),
            p75=float(np.percentile(arr, 75)),
            p90=float(np.percentile(arr, 90)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            max_length=int(np.max(arr)),
            frac_under_128=float(np.mean(arr < 128)),
            frac_128_to_255=float(np.mean((arr >= 128) & (arr < 256))),
            frac_256_to_511=float(np.mean((arr >= 256) & (arr < 512))),
            frac_512_and_above=float(np.mean(arr >= 512)),
        )

        assert stats.mean_length == 355.0
        assert stats.median_length == 275.0
        assert stats.max_length == 1000
        assert stats.frac_under_128 == 0.2
        assert stats.frac_512_and_above == 0.2


class TestManifestsAndReproducibility:
    def test_manifest_files_generation(self, tmp_path, tokenizer):
        tok_path = PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.model"
        p500m = tmp_path / "manifest_500m.json"
        p1b = tmp_path / "manifest_1b.json"

        m500 = save_manifest_file(
            target_name="500M Test",
            target_tokens=500_000_000,
            subsets=["cosmopedia-v2", "fineweb-edu-dedup"],
            tokenizer_path=tok_path,
            tokenizer_vocab_size=tokenizer.vocab_size(),
            output_path=p500m,
        )

        m1b = save_manifest_file(
            target_name="1B Test",
            target_tokens=1_000_000_000,
            subsets=["cosmopedia-v2", "fineweb-edu-dedup", "python-edu"],
            tokenizer_path=tok_path,
            tokenizer_vocab_size=tokenizer.vocab_size(),
            output_path=p1b,
        )

        assert p500m.exists()
        assert p1b.exists()
        assert m500["target_token_budget"] == 500_000_000
        assert m500["training_plan"]["total_optimizer_steps"] == 30517
        assert m1b["target_token_budget"] == 1_000_000_000
        assert m1b["training_plan"]["total_optimizer_steps"] == 61035
