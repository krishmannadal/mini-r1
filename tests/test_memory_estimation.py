"""
Mini-R1 -- Memory Estimation Unit Tests
=======================================
Tests verifying:
1. Exact analytical parameter counting matches PyTorch parameter counts.
2. Weight, gradient, and AdamW optimizer memory equations.
3. Activation memory scaling across batch sizes.
4. Total memory and VRAM threshold checks.
"""

from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.model import MiniR1Config, MiniR1Transformer
from src.memory import (
    calculate_parameter_breakdown,
    estimate_training_memory,
    ParameterBreakdown,
    MemoryEstimation,
)


class TestMemoryEstimation:
    """Verify memory computation accuracy."""

    def test_parameter_breakdown_matches_pytorch_model(self):
        """Analytical parameter count must match PyTorch model.parameters() exactly."""
        config = MiniR1Config(
            vocab_size=8000,
            d_model=512,
            n_layers=12,
            n_heads=8,
            d_ff=2048,
            max_seq_len=512,
            bias=False,
            tie_word_embeddings=True,
        )
        model = MiniR1Transformer(config)
        pytorch_params = sum(p.numel() for p in model.parameters())

        breakdown: ParameterBreakdown = calculate_parameter_breakdown(config)

        assert breakdown.total_params == 42_132_480
        assert breakdown.total_params == pytorch_params
        assert breakdown.non_embedding_params == 42_132_480 - (8000 * 512) - (512 * 512)

    def test_static_memory_formulas(self):
        """Weights, gradients, and optimizer memory must follow byte formulas."""
        config = MiniR1Config(
            vocab_size=8000,
            d_model=512,
            n_layers=12,
            n_heads=8,
            d_ff=2048,
            max_seq_len=512,
        )
        est: MemoryEstimation = estimate_training_memory(config, batch_size=8, dtype_bytes=4)

        expected_weights_bytes = 42_132_480 * 4
        expected_grads_bytes = 42_132_480 * 4
        expected_optim_bytes = 42_132_480 * 8  # 2 states (m, v) in FP32
        expected_static_bytes = expected_weights_bytes + expected_grads_bytes + expected_optim_bytes

        assert est.weights_bytes == expected_weights_bytes
        assert est.gradients_bytes == expected_grads_bytes
        assert est.optimizer_bytes == expected_optim_bytes
        assert est.static_bytes == expected_static_bytes

    def test_activation_memory_scales_with_batch_size(self):
        """Activations must scale linearly with batch size."""
        config = MiniR1Config(d_model=512, n_layers=12, max_seq_len=512)

        est_b4 = estimate_training_memory(config, batch_size=4)
        est_b8 = estimate_training_memory(config, batch_size=8)
        est_b16 = estimate_training_memory(config, batch_size=16)

        assert est_b8.activations_bytes == est_b4.activations_bytes * 2
        assert est_b16.activations_bytes == est_b8.activations_bytes * 2

    def test_vram_fit_thresholds(self):
        """At 6 GB VRAM, B=8 must fit comfortably, and high batch sizes flag OOM."""
        config = MiniR1Config(d_model=512, n_layers=12, max_seq_len=512)

        est_b8 = estimate_training_memory(config, batch_size=8, vram_budget_gb=6.0)
        assert est_b8.fits_in_vram is True
        assert est_b8.total_training_gb < 3.5

        est_b32 = estimate_training_memory(config, batch_size=32, vram_budget_gb=6.0)
        assert est_b32.fits_in_vram is False
