"""
Mini-R1 -- Transformer Block Module
===================================
Implements a standard Pre-LayerNorm Transformer Decoder Block:
    1. Multi-Head Causal Self-Attention with Residual Connection
    2. Pointwise Feed-Forward Network with Residual Connection

Mathematical formulation:
    h = x + MultiHeadAttention(LayerNorm1(x))
    out = h + FeedForward(LayerNorm2(h))
"""

from __future__ import annotations

from typing import NamedTuple
import torch
import torch.nn as nn

from src.attention import MultiHeadCausalSelfAttention, AttentionOutputs


class BlockOutputs(NamedTuple):
    """Container exposing intermediate block outputs for inspection."""
    output: torch.Tensor
    attn_intermediates: AttentionOutputs | None


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN).

    Mathematical formulation:
        FFN(x) = GELU(x @ W_1 + b_1) @ W_2 + b_2
    """

    def __init__(
        self,
        d_model: int = 128,
        d_ff: int = 512,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        # Expansion layer: d_model -> d_ff (e.g. 128 -> 512)
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.activation = nn.GELU()
        # Projection layer: d_ff -> d_model (e.g. 512 -> 128)
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, d_model)
        Returns:
            Tensor of shape (B, T, d_model)
        """
        h = self.fc1(x)
        h = self.activation(h)
        h = self.fc2(h)
        return self.dropout(h)


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Decoder Transformer Block.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 32,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        # Pre-LayerNorm for Attention sublayer
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            bias=bias,
        )

        # Pre-LayerNorm for FFN sublayer
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor | BlockOutputs:
        """
        Forward pass for Transformer Block.

        Args:
            x: Input tensor of shape (B, T, d_model)
            return_intermediates: If True, returns BlockOutputs containing attention internals.

        Returns:
            Tensor of shape (B, T, d_model) or BlockOutputs namedtuple.
        """
        # 1. Self-Attention with Pre-LN and Residual connection
        ln1_out = self.ln1(x)
        if return_intermediates:
            attn_res: AttentionOutputs = self.attn(ln1_out, return_intermediates=True)
            x = x + attn_res.output
        else:
            attn_out = self.attn(ln1_out, return_intermediates=False)
            x = x + attn_out
            attn_res = None

        # 2. Feed-Forward Network with Pre-LN and Residual connection
        ln2_out = self.ln2(x)
        ffn_out = self.ffn(ln2_out)
        x = x + ffn_out

        if return_intermediates:
            return BlockOutputs(output=x, attn_intermediates=attn_res)

        return x
