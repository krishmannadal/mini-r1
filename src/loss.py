"""
Mini-R1 -- Loss Computation Module
==================================
Implements causal next-token cross-entropy loss with teacher forcing
and label shifting.

Mathematical formulation:
    Given logits of shape (B, T, V) and targets of shape (B, T):
    1. Shift logits:  logits[:, :-1, :] -> (B, T-1, V)
    2. Shift targets: targets[:, 1:]    -> (B, T-1)
    3. Flatten:
       logits_flat  = shift_logits.reshape(-1, V)
       targets_flat = shift_targets.reshape(-1)
    4. Cross-Entropy Loss:
       loss = CrossEntropy(logits_flat, targets_flat, ignore_index=-100)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_causal_lm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute next-token prediction cross-entropy loss with teacher forcing shifting.

    Args:
        logits: Unnormalized model outputs of shape (B, T, V)
        targets: Target token IDs of shape (B, T)
        ignore_index: Target index to ignore in cross-entropy calculation (default: -100)

    Returns:
        Scalar cross-entropy loss tensor.
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected 3D logits of shape (B, T, V), got shape {logits.shape}")
    if targets.ndim != 2:
        raise ValueError(f"Expected 2D targets of shape (B, T), got shape {targets.shape}")

    batch_size, seq_len, vocab_size = logits.shape
    t_batch, t_len = targets.shape

    if batch_size != t_batch or seq_len != t_len:
        raise ValueError(
            f"Logits shape ({batch_size}, {seq_len}, {vocab_size}) incompatible "
            f"with targets shape ({t_batch}, {t_len})"
        )

    if seq_len < 2:
        raise ValueError(f"Sequence length must be >= 2 for next-token prediction, got {seq_len}")

    # 1. Shift: model at position t predicts token at position t+1
    shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
    shift_targets = targets[:, 1:].contiguous()     # (B, T-1)

    # 2. Flatten for PyTorch cross-entropy
    flat_logits = shift_logits.view(-1, vocab_size)  # (B * (T-1), V)
    flat_targets = shift_targets.view(-1)            # (B * (T-1))

    # 3. Compute loss
    loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index)
    return loss
