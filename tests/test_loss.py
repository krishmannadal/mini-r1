"""
Mini-R1 -- Loss Computation Test Suite
======================================
Tests verifying:
1. Next-token loss computation and teacher-forcing target shifting.
2. Loss scalar output shape.
3. Ignore index masking (-100).
4. Error handling on incompatible shapes.
"""

from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.loss import compute_causal_lm_loss


class TestCausalLoss:
    """Verify loss function mathematics and shifting."""

    def test_loss_returns_scalar(self):
        """Loss must be a 0-dimensional scalar tensor."""
        B, T, V = 2, 8, 100
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))

        loss = compute_causal_lm_loss(logits, targets)
        assert loss.ndim == 0
        assert loss.item() > 0.0

    def test_teacher_forcing_shift_logic(self):
        """
        Verify that logit at position t predicts target at position t+1.
        Construct an artificial scenario where logits[:, :-1, :] perfectly predict targets[:, 1:].
        """
        B, T, V = 1, 4, 10
        targets = torch.tensor([[1, 5, 8, 3]], dtype=torch.long)
        # Shifted targets are [5, 8, 3] at positions 0, 1, 2
        logits = torch.zeros(B, T, V)
        # Set large positive logit for the correct next token
        logits[0, 0, 5] = 100.0  # pos 0 predicts 5 (targets[1])
        logits[0, 1, 8] = 100.0  # pos 1 predicts 8 (targets[2])
        logits[0, 2, 3] = 100.0  # pos 2 predicts 3 (targets[3])

        loss = compute_causal_lm_loss(logits, targets)
        # Perfect prediction should yield loss very close to 0.0
        assert loss.item() < 1e-4

    def test_ignore_index_masking(self):
        """Tokens with ignore_index=-100 must not contribute to cross-entropy loss."""
        B, T, V = 1, 4, 10
        logits = torch.randn(B, T, V)
        targets = torch.tensor([[1, -100, -100, 3]], dtype=torch.long)

        loss = compute_causal_lm_loss(logits, targets, ignore_index=-100)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_incompatible_shapes_raise_error(self):
        """Mismatch in dimensions must raise ValueError."""
        logits = torch.randn(2, 8, 100)
        targets_wrong = torch.randint(0, 100, (2, 6))

        with pytest.raises(ValueError, match="incompatible"):
            compute_causal_lm_loss(logits, targets_wrong)
