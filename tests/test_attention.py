"""
Mini-R1 -- Causal Self-Attention Test Suite
===========================================
Tests verifying:
1. Multi-Head Attention output shapes.
2. Q, K, V intermediate tensor dimensions.
3. Attention matrix dimensions (B, n_heads, T, T).
4. Causal mask enforcement (strictly zero attention to future tokens j > i).
5. Row normalization (attention weights sum to 1.0 for each query position).
"""

from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.attention import MultiHeadCausalSelfAttention, AttentionOutputs


class TestAttentionShapes:
    """Verify output and intermediate tensor dimensions."""

    def test_attention_output_shape(self):
        """Attention output must match input shape (B, T, d_model)."""
        B, T, d_model, n_heads = 2, 8, 128, 4
        attn = MultiHeadCausalSelfAttention(d_model=d_model, n_heads=n_heads, max_seq_len=32)
        x = torch.randn(B, T, d_model)

        out = attn(x)
        assert out.shape == (B, T, d_model)

    def test_qkv_and_matrix_shapes(self):
        """Q, K, V must be (B, n_heads, T, d_k) and matrix must be (B, n_heads, T, T)."""
        B, T, d_model, n_heads = 2, 6, 128, 4
        d_k = d_model // n_heads  # 32
        attn = MultiHeadCausalSelfAttention(d_model=d_model, n_heads=n_heads, max_seq_len=32)
        x = torch.randn(B, T, d_model)

        res: AttentionOutputs = attn(x, return_intermediates=True)

        assert res.output.shape == (B, T, d_model)
        assert res.q.shape == (B, n_heads, T, d_k)
        assert res.k.shape == (B, n_heads, T, d_k)
        assert res.v.shape == (B, n_heads, T, d_k)
        assert res.attn_matrix.shape == (B, n_heads, T, T)


class TestCausalMasking:
    """Verify causal attention semantics."""

    def test_causal_mask_zeros_future_tokens(self):
        """Upper triangular elements (j > i) in the attention matrix must be strictly 0.0."""
        B, T, d_model, n_heads = 2, 10, 128, 4
        attn = MultiHeadCausalSelfAttention(d_model=d_model, n_heads=n_heads, max_seq_len=32)
        x = torch.randn(B, T, d_model)

        res: AttentionOutputs = attn(x, return_intermediates=True)
        attn_mat = res.attn_matrix  # (B, n_heads, T, T)

        for b in range(B):
            for h in range(n_heads):
                mat = attn_mat[b, h]
                for i in range(T):
                    for j in range(T):
                        if j > i:
                            assert mat[i, j].item() == 0.0, f"Future leakage at pos ({i}, {j}): {mat[i, j].item()}"

    def test_attention_rows_sum_to_one(self):
        """Each row of the attention matrix must sum to 1.0."""
        B, T, d_model, n_heads = 2, 8, 128, 4
        attn = MultiHeadCausalSelfAttention(d_model=d_model, n_heads=n_heads, max_seq_len=32)
        x = torch.randn(B, T, d_model)

        res: AttentionOutputs = attn(x, return_intermediates=True)
        row_sums = res.attn_matrix.sum(dim=-1)  # (B, n_heads, T)

        expected = torch.ones_like(row_sums)
        assert torch.allclose(row_sums, expected, atol=1e-5)
