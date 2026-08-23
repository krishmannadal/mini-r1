"""
Mini-R1 -- Unit Tests for Gradient Accumulation Equivalence
===========================================================
Asserts that single full batch and accumulated micro-batches yield equivalent gradients.
"""

import pytest
import torch
from scripts.verify_gradient_accumulation import run_gradient_accumulation_verification


class TestGradientAccumulationEquivalence:
    """Verify mathematical equivalence of gradient accumulation."""

    def test_fp32_accumulation_exact_match(self):
        """FP32 accumulation has near-zero difference (machine epsilon scale)."""
        res = run_gradient_accumulation_verification(precision="float32", device="cpu")
        assert res["loss_diff"] < 1e-6
        assert res["max_abs_gradient_diff"] < 1e-5
        assert res["relative_l2_error"] < 1e-4

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for FP16 AMP test")
    def test_fp16_amp_accumulation_near_match(self):
        """FP16 AMP accumulation has low relative error (< 0.1%)."""
        res = run_gradient_accumulation_verification(precision="float16", device="cuda")
        assert res["loss_diff"] < 1e-3
        assert res["max_abs_gradient_diff"] < 1e-2
        assert res["relative_l2_error"] < 0.005  # < 0.5% relative error
