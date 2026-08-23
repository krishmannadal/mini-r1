"""
Mini-R1 -- Optimizer Configuration Module
=========================================
Configures AdamW optimizer with decoupled weight decay parameter grouping.

Decoupled Weight Decay Strategy:
    - 2D tensor parameters (e.g. projection weights, linear weights) receive weight decay.
    - 1D tensor parameters (e.g. biases, LayerNorm scale and shift) receive weight decay = 0.0.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def create_adamw_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """
    Create an AdamW optimizer instance with decoupled weight decay parameter grouping.

    Args:
        model: PyTorch nn.Module instance
        lr: Learning rate (default: 1e-3)
        weight_decay: L2 penalty factor for 2D weight matrices (default: 0.01)
        betas: Adam momentum coefficients (default: (0.9, 0.95))
        eps: Numerical stability epsilon (default: 1e-8)

    Returns:
        Configured torch.optim.AdamW optimizer.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 2D+ tensors (weight matrices) decay; 1D tensors (biases, LayerNorm scale) do not decay
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=lr,
        betas=betas,
        eps=eps,
    )
    return optimizer
