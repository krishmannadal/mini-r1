"""
Mini-R1 -- Multi-Head Causal Self-Attention Module
==================================================
Implemented entirely from scratch using basic PyTorch tensor operations.
(Zero usage of nn.MultiheadAttention).

Mathematical formulation:
    1. Linear Projections:
       Q = X @ W_q,  K = X @ W_k,  V = X @ W_v
    2. Multi-Head Reshape:
       (B, T, d_model) -> (B, n_heads, T, d_k) where d_k = d_model / n_heads
    3. Scaled Dot-Product:
       Scores = (Q @ K^T) / sqrt(d_k) -> (B, n_heads, T, T)
    4. Causal Mask:
       Mask[i, j] = -inf for j > i
    5. Softmax:
       A = softmax(Scores + Mask, dim=-1) -> (B, n_heads, T, T)
    6. Context Aggregation:
       Out = A @ V -> (B, n_heads, T, d_k)
    7. Head Concatenation & Output Projection:
       Out = Reshape(Out) @ W_o -> (B, T, d_model)
"""

from __future__ import annotations

import math
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionOutputs(NamedTuple):
    """Container exposing intermediate attention tensors for inspection."""
    output: torch.Tensor           # (B, T, d_model)
    q: torch.Tensor                # (B, n_heads, T, d_k)
    k: torch.Tensor                # (B, n_heads, T, d_k)
    v: torch.Tensor                # (B, n_heads, T, d_k)
    attn_matrix: torch.Tensor      # (B, n_heads, T, T)


class MultiHeadCausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention from scratch.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        max_seq_len: int = 32,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head (e.g. 128 / 4 = 32)
        self.max_seq_len = max_seq_len

        # 1. Explicit Linear Projections for Query, Key, and Value
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        # 2. Output Projection after head concatenation
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # 3. Dropout
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # 4. Lower-triangular causal mask buffer (1s on and below diagonal, 0s above)
        # Register as non-persistent buffer so it moves with the model across devices
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor | AttentionOutputs:
        """
        Forward pass for causal self-attention.

        Args:
            x: Input tensor of shape (B, T, d_model)
            return_intermediates: If True, returns AttentionOutputs with Q, K, V, and attention matrix.

        Returns:
            Tensor of shape (B, T, d_model) or AttentionOutputs namedtuple.
        """
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model, f"Input feature dim {d_model} does not match model dim {self.d_model}"

        # ---------------------------------------------------------------------
        # Step 1: Linear projections -> (B, T, d_model)
        # ---------------------------------------------------------------------
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ---------------------------------------------------------------------
        # Step 2: Split into multiple heads -> (B, n_heads, T, d_k)
        # Transpose from (B, T, n_heads, d_k) to (B, n_heads, T, d_k)
        # ---------------------------------------------------------------------
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # ---------------------------------------------------------------------
        # Step 3: Scaled Dot-Product -> (B, n_heads, T, T)
        # Matrix multiplication between queries and transposed keys
        # ---------------------------------------------------------------------
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # ---------------------------------------------------------------------
        # Step 4: Apply Causal Mask
        # Upper triangle (j > i) filled with -inf so softmax(scores) becomes 0.0
        # ---------------------------------------------------------------------
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(~mask, float("-inf"))

        # ---------------------------------------------------------------------
        # Step 5: Softmax over key positions (dim=-1) -> (B, n_heads, T, T)
        # ---------------------------------------------------------------------
        attn_matrix = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_matrix)

        # ---------------------------------------------------------------------
        # Step 6: Weighted Context Aggregation -> (B, n_heads, T, d_k)
        # ---------------------------------------------------------------------
        context = torch.matmul(attn_weights, v)

        # ---------------------------------------------------------------------
        # Step 7: Concatenate heads back to (B, T, d_model) & Output Projection
        # ---------------------------------------------------------------------
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(context)
        out = self.resid_dropout(out)

        if return_intermediates:
            return AttentionOutputs(
                output=out,
                q=q,
                k=k,
                v=v,
                attn_matrix=attn_matrix,
            )

        return out
