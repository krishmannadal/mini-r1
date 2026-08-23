"""
Mini-R1 -- DataLoader and Collate Module
========================================
Implements custom collation with dynamic batch padding and cross-entropy target
masking (-100 for pad tokens).
"""

from __future__ import annotations

from typing import NamedTuple, Sequence
import torch
from torch.utils.data import DataLoader, Dataset


class BatchData(NamedTuple):
    """Container holding batched tensors for model training."""
    input_ids: torch.Tensor       # (B, T)
    targets: torch.Tensor         # (B, T) with -100 on pad positions
    attention_mask: torch.Tensor  # (B, T) with 1 for real tokens, 0 for pad


class CausalCollateFn:
    """
    Collate function that dynamically pads sequences in the batch and creates
    targets with -100 label masking on pad tokens.
    """

    def __init__(self, pad_id: int = 3, ignore_index: int = -100) -> None:
        self.pad_id = pad_id
        self.ignore_index = ignore_index

    def __call__(self, batch: Sequence[torch.Tensor]) -> BatchData:
        """
        Pad variable-length sequence tensors into a uniform 2D batch.

        Args:
            batch: List of 1D LongTensors

        Returns:
            BatchData containing input_ids, targets, and attention_mask.
        """
        batch_size = len(batch)
        max_len = max(len(seq) for seq in batch)

        # Initialize input_ids with pad_id
        input_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        # Initialize targets with ignore_index (-100)
        targets = torch.full((batch_size, max_len), self.ignore_index, dtype=torch.long)
        # Initialize attention_mask with 0
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, seq in enumerate(batch):
            seq_len = len(seq)
            input_ids[i, :seq_len] = seq
            # Targets for real tokens match input_ids; pad positions remain -100
            targets[i, :seq_len] = seq
            attention_mask[i, :seq_len] = 1

        return BatchData(
            input_ids=input_ids,
            targets=targets,
            attention_mask=attention_mask,
        )


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    pad_id: int = 3,
    ignore_index: int = -100,
    num_workers: int = 0,
) -> DataLoader:
    """
    Factory function creating a PyTorch DataLoader configured for causal language modeling.
    """
    collate_fn = CausalCollateFn(pad_id=pad_id, ignore_index=ignore_index)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=False,
    )
