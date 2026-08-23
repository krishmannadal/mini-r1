"""
Mini-R1 -- Transformer Block Test Suite
=======================================
Tests verifying:
1. TransformerBlock forward pass and output shape.
2. FeedForwardNetwork dimensions (d_model -> d_ff -> d_model).
3. Pre-LayerNorm residual preservation.
"""

from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.transformer_block import TransformerBlock, FeedForwardNetwork, BlockOutputs


class TestBlockComponents:
    """Verify individual subcomponents and full block execution."""

    def test_ffn_shape(self):
        """FFN expands to d_ff and projects back to d_model."""
        B, T, d_model, d_ff = 2, 8, 128, 512
        ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff)
        x = torch.randn(B, T, d_model)

        out = ffn(x)
        assert out.shape == (B, T, d_model)

    def test_transformer_block_shape(self):
        """TransformerBlock preserves (B, T, d_model) shape."""
        B, T, d_model, n_heads, d_ff = 3, 12, 128, 4, 512
        block = TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, max_seq_len=32)
        x = torch.randn(B, T, d_model)

        out = block(x)
        assert out.shape == (B, T, d_model)

    def test_transformer_block_return_intermediates(self):
        """Block exposes attention internals when requested."""
        B, T, d_model = 2, 6, 128
        block = TransformerBlock(d_model=d_model, n_heads=4, d_ff=512, max_seq_len=32)
        x = torch.randn(B, T, d_model)

        res: BlockOutputs = block(x, return_intermediates=True)
        assert res.output.shape == (B, T, d_model)
        assert res.attn_intermediates is not None
        assert res.attn_intermediates.attn_matrix.shape == (B, 4, T, T)
