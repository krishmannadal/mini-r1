"""
Mini-R1 -- Training Step Test Suite
===================================
Tests verifying:
1. Gradients exist and are non-zero after backward pass.
2. Parameters change after optimizer step.
3. Loss strictly decreases after optimization on a fixed batch.
4. AdamW optimizer parameter groups.
"""

from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.model import MiniR1Transformer, MiniR1Config
from src.optimizer import create_adamw_optimizer
from src.training_step import perform_training_step, compute_gradient_norm
from src.loss import compute_causal_lm_loss


class TestTrainingStep:
    """Verify backpropagation and parameter update mechanics."""

    def test_gradients_exist_and_non_zero(self):
        """All trainable parameters must have non-zero gradients after backward pass."""
        config = MiniR1Config(vocab_size=100, d_model=32, n_layers=1, n_heads=2, d_ff=64, max_seq_len=16)
        model = MiniR1Transformer(config)
        optimizer = create_adamw_optimizer(model, lr=1e-3)

        input_ids = torch.randint(0, 100, (2, 8))
        res = perform_training_step(model, optimizer, input_ids)

        assert res.loss > 0.0
        assert res.grad_norm > 0.0
        # Check that individual parameter gradients exist
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Parameter {name} missing gradient"
                assert name in res.param_gradients

    def test_parameters_change_after_step(self):
        """Weights must change value after optimizer.step()."""
        config = MiniR1Config(vocab_size=100, d_model=32, n_layers=1, n_heads=2, d_ff=64, max_seq_len=16)
        model = MiniR1Transformer(config)
        optimizer = create_adamw_optimizer(model, lr=1e-2)

        input_ids = torch.randint(0, 100, (2, 8))
        target_param_name = "blocks.0.attn.q_proj.weight"
        before = model.blocks[0].attn.q_proj.weight.data.clone()

        perform_training_step(model, optimizer, input_ids)

        after = model.blocks[0].attn.q_proj.weight.data
        delta = (after - before).norm().item()
        assert delta > 0.0, "Parameter did not update after training step"

    def test_loss_decreases_after_training_step(self):
        """Recomputed loss on the same batch must strictly decrease after optimizer update."""
        torch.manual_seed(42)
        config = MiniR1Config(vocab_size=200, d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq_len=16)
        model = MiniR1Transformer(config)
        optimizer = create_adamw_optimizer(model, lr=5e-3)

        input_ids = torch.randint(0, 200, (2, 10))

        # Initial loss
        model.eval()
        with torch.no_grad():
            init_loss = compute_causal_lm_loss(model(input_ids), input_ids).item()

        # Perform 1 optimization step
        perform_training_step(model, optimizer, input_ids)

        # Recomputed loss
        model.eval()
        with torch.no_grad():
            updated_loss = compute_causal_lm_loss(model(input_ids), input_ids).item()

        assert updated_loss < init_loss, f"Loss did not decrease: {init_loss:.4f} -> {updated_loss:.4f}"

    def test_adamw_param_groups(self):
        """AdamW separates 2D weights (decay) from 1D biases/norms (no decay)."""
        config = MiniR1Config(vocab_size=100, d_model=32, n_layers=1, n_heads=2, d_ff=64)
        model = MiniR1Transformer(config)
        optimizer = create_adamw_optimizer(model, weight_decay=0.05)

        assert len(optimizer.param_groups) == 2
        decay_group = optimizer.param_groups[0]
        no_decay_group = optimizer.param_groups[1]

        assert decay_group["weight_decay"] == 0.05
        assert no_decay_group["weight_decay"] == 0.0
