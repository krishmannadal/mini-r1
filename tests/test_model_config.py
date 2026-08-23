"""
Mini-R1 -- Model Configuration & Scaling Unit Tests
===================================================
Tests verifying:
1. Loading configuration from YAML.
2. Serialization to and from dictionary.
3. Attention head dimension calculation (d_k).
4. Architecture B parameter counts (42.1M parameters).
5. Forward pass with the 42M parameter configuration.
"""

from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.model import MiniR1Config, MiniR1Transformer


class TestModelConfiguration:
    """Verify configuration loading and validation."""

    def test_load_from_yaml(self):
        """MiniR1Config loads accurately from configs/model.yaml."""
        yaml_path = PROJECT_ROOT / "configs" / "model.yaml"
        assert yaml_path.exists(), "configs/model.yaml must exist"

        config = MiniR1Config.from_yaml(yaml_path)
        assert config.vocab_size == 8000
        assert config.d_model == 512
        assert config.n_layers == 12
        assert config.n_heads == 8
        assert config.d_ff == 2048
        assert config.max_seq_len == 512
        assert config.bias is False
        assert config.tie_word_embeddings is True

    def test_head_dimension_property(self):
        """d_k calculates d_model // n_heads correctly."""
        config = MiniR1Config(d_model=512, n_heads=8)
        assert config.d_k == 64

        invalid_config = MiniR1Config(d_model=512, n_heads=7)
        with pytest.raises(AssertionError):
            _ = invalid_config.d_k

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict and from_dict preserve all hyperparameters."""
        original = MiniR1Config(
            vocab_size=8000,
            d_model=512,
            n_layers=12,
            n_heads=8,
            d_ff=2048,
            max_seq_len=512,
            dropout=0.1,
            bias=True,
            tie_word_embeddings=False,
        )
        d = original.to_dict()
        restored = MiniR1Config.from_dict(d)

        assert original.vocab_size == restored.vocab_size
        assert original.d_model == restored.d_model
        assert original.n_layers == restored.n_layers
        assert original.n_heads == restored.n_heads
        assert original.d_ff == restored.d_ff
        assert original.max_seq_len == restored.max_seq_len
        assert original.dropout == restored.dropout
        assert original.bias == restored.bias
        assert original.tie_word_embeddings == restored.tie_word_embeddings

    def test_architecture_b_parameter_count(self):
        """Architecture B (512-dim, 12-layer, 8-head) has exactly ~42.1M parameters."""
        config = MiniR1Config.from_yaml(PROJECT_ROOT / "configs" / "model.yaml")
        model = MiniR1Transformer(config)

        total_params = sum(p.numel() for p in model.parameters())
        # Theoretical exact count: 42,132,480
        assert total_params == 42_132_480, f"Expected 42,132,480 parameters, got {total_params:,}"

    def test_architecture_b_forward_pass(self):
        """Model forward pass operates correctly on scaled configuration."""
        config = MiniR1Config(
            vocab_size=8000,
            d_model=512,
            n_layers=2,  # Use 2 layers for fast unit test execution
            n_heads=8,
            d_ff=2048,
            max_seq_len=64,
        )
        model = MiniR1Transformer(config)
        model.eval()

        B, T = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, T))
        with torch.no_grad():
            logits = model(input_ids)

        assert logits.shape == (B, T, config.vocab_size)
