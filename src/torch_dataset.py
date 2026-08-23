"""
Mini-R1 -- PyTorch Dataset Wrapper Module
=========================================
Wraps formatted reasoning samples into a standard PyTorch Dataset.
Encodes text into token ID tensors with BOS and EOS delimiters and enforces max_seq_len truncation.
"""

from __future__ import annotations

from typing import Any, Sequence
import torch
from torch.utils.data import Dataset

from src.tokenizer import Tokenizer


class ReasoningTorchDataset(Dataset):
    """
    PyTorch Dataset wrapping reasoning records (GSM8K, SVAMP, AQuA).
    """

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        tokenizer: Tokenizer,
        max_seq_len: int = 32,
    ) -> None:
        """
        Args:
            records: List of formatted sample dictionaries containing 'formatted_text'
            tokenizer: Initialized Mini-R1 Tokenizer instance
            max_seq_len: Maximum token length (default: 32)
        """
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Encode a single sample into a 1D LongTensor with BOS and EOS tokens.

        Returns:
            1D LongTensor of token IDs (shape: (L,) where L <= max_seq_len)
        """
        item = self.records[idx]
        text = item["formatted_text"]

        # Encode with BOS and EOS tokens
        token_ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)

        # Truncate if exceeds max_seq_len
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len]
            # Ensure the sequence ends with EOS if truncated
            token_ids[-1] = self.tokenizer.eos_id

        return torch.tensor(token_ids, dtype=torch.long)
