"""
Mini-R1 -- Training Step Module
===============================
Encapsulates a single atomic optimization step for the Transformer model:
1. Zero gradients
2. Forward pass (logits generation)
3. Cross-entropy loss computation
4. Backward pass (autograd gradient accumulation)
5. Gradient norm inspection and clipping
6. Parameter update via optimizer step
"""

from __future__ import annotations

from typing import NamedTuple
import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.loss import compute_causal_lm_loss


class TrainingStepResult(NamedTuple):
    """Container holding loss metrics and gradient statistics from a training step."""
    loss: float
    grad_norm: float
    param_gradients: dict[str, float]


def compute_gradient_norm(model: nn.Module) -> tuple[float, dict[str, float]]:
    """
    Calculate the global Euclidean gradient norm and individual layer norms.

    Returns:
        total_norm: Global L2 norm across all parameters.
        param_norms: Dictionary of L2 gradient norms per named parameter.
    """
    total_norm_sq = 0.0
    param_norms = {}

    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            param_norms[name] = param_norm
            total_norm_sq += param_norm ** 2

    total_norm = total_norm_sq ** 0.5
    return total_norm, param_norms


def perform_training_step(
    model: nn.Module,
    optimizer: Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor | None = None,
    max_grad_norm: float | None = 1.0,
    ignore_index: int = -100,
) -> TrainingStepResult:
    """
    Execute one complete forward pass, loss calculation, backward pass,
    gradient clipping, and optimizer parameter update.

    Args:
        model: Transformer model instance
        optimizer: Configured optimizer instance (e.g. AdamW)
        input_ids: Tensor of input token IDs with shape (B, T)
        targets: Optional target token IDs with shape (B, T) (defaults to input_ids)
        max_grad_norm: Max gradient norm for clipping (default: 1.0, None to disable)
        ignore_index: Index to ignore in cross-entropy loss (default: -100)

    Returns:
        TrainingStepResult containing scalar loss and gradient norms.
    """
    model.train()
    optimizer.zero_grad()

    if targets is None:
        targets = input_ids

    # 1. Forward Pass -> (B, T, vocab_size)
    logits = model(input_ids)

    # 2. Loss Computation
    loss = compute_causal_lm_loss(logits, targets, ignore_index=ignore_index)

    # 3. Backward Pass (Accumulates gradients)
    loss.backward()

    # 4. Inspect Gradients (Pre-clip norm)
    total_grad_norm, param_norms = compute_gradient_norm(model)

    # 5. Gradient Clipping (if requested)
    if max_grad_norm is not None and max_grad_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # 6. Optimizer Step (Update parameters)
    optimizer.step()

    return TrainingStepResult(
        loss=loss.item(),
        grad_norm=total_grad_norm,
        param_gradients=param_norms,
    )
