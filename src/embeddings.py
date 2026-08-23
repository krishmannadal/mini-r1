"""
Mini-R1 -- Embedding Layer Module
=================================
Combines learned Token Embeddings and learned 1D Positional Embeddings.

Shapes:
    Input:
        input_ids: LongTensor of shape (B, T) where T <= max_seq_len
    Output:
        x: FloatTensor of shape (B, T, d_model)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerEmbedding(nn.Module):
    """
    Combined Token + Learned 1D Positional Embeddings.

    Mathematical formulation:
        x = TokenEmbedding(input_ids) + PositionalEmbedding(positions)
    """

    def __init__(
        self,
        vocab_size: int = 8000,
        d_model: int = 128,
        max_seq_len: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding lookup table W_tok in R^(vocab_size x d_model)
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # Learned positional embedding lookup table W_pos in R^(max_seq_len x d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for embedding layer.

        Args:
            input_ids: LongTensor of shape (B, T)

        Returns:
            Tensor of shape (B, T, d_model)
        """
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum sequence length {self.max_seq_len}"
            )

        # 1. Lookup token embeddings -> (B, T, d_model)
        tok_out = self.tok_emb(input_ids)

        # 2. Generate position indices [0, 1, ..., seq_len - 1] on the same device
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)

        # 3. Lookup positional embeddings -> (T, d_model)
        pos_out = self.pos_emb(positions)

        # 4. Elementwise addition with broadcasting: (B, T, d_model) + (T, d_model) -> (B, T, d_model)
        x = tok_out + pos_out
        return self.drop(x)
