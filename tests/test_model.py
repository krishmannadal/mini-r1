"""
Mini-R1 -- Model Test Suite
===========================
Tests verifying:
1. MiniR1Transformer forward pass and logits shape (B, T, vocab_size).
2. Max sequence length bounds checking.
3. Multiple batch size execution.
4. Parameter counting and weight tying.
"""

from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.model import MiniR1Transformer, MiniR1Config, ModelForwardOutputs


class TestMiniR1Transformer:
    """Verify full decoder model forward pass and boundaries."""

    def test_forward_logits_shape(self):
        """Model forward pass must output (B, T, vocab_size)."""
        config = MiniR1Config(
            vocab_size=8000,
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            max_seq_len=32,
        )
        model = MiniR1Transformer(config)
        model.eval()

        B, T = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, T))
        logits = model(input_ids)

        assert logits.shape == (B, T, config.vocab_size)

    def test_forward_with_intermediates(self):
        """Model forward pass returns all block traces when requested."""
        config = MiniR1Config(
            vocab_size=8000,
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            max_seq_len=32,
        )
        model = MiniR1Transformer(config)
        model.eval()

        B, T = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (B, T))
        outputs: ModelForwardOutputs = model(input_ids, return_intermediates=True)

        assert outputs.logits.shape == (B, T, config.vocab_size)
        assert outputs.embedding_out.shape == (B, T, config.d_model)
        assert len(outputs.block_outputs) == config.n_layers
        assert outputs.block_outputs[0].output.shape == (B, T, config.d_model)

    def test_max_seq_len_exceeded_raises_error(self):
        """Passing sequence length > max_seq_len (32) must raise ValueError."""
        config = MiniR1Config(max_seq_len=32)
        model = MiniR1Transformer(config)

        invalid_ids = torch.randint(0, config.vocab_size, (1, 33))
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            model(invalid_ids)

    def test_weight_tying(self):
        """When tie_word_embeddings is True, lm_head and tok_emb share weights."""
        config = MiniR1Config(tie_word_embeddings=True)
        model = MiniR1Transformer(config)

        assert model.lm_head.weight is model.embeddings.tok_emb.weight

    def test_parameter_count(self):
        """Verify model parameter count computation."""
        config = MiniR1Config(vocab_size=8000, d_model=128, n_layers=2, d_ff=512)
        model = MiniR1Transformer(config)

        total_params = model.get_num_params()
        assert total_params > 1_000_000, f"Expected >1M params, got {total_params}"
