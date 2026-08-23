"""
Mini-R1 -- Checkpointing Test Suite
===================================
Tests verifying:
1. save_checkpoint serializes model, optimizer, step count, and metadata.
2. load_checkpoint restores exact parameter weights and metadata.
3. resume_training recovers correct start epoch and global step.
"""

from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.model import MiniR1Transformer, MiniR1Config
from src.optimizer import create_adamw_optimizer
from src.checkpoint import save_checkpoint, load_checkpoint, resume_training


class TestCheckpoint:
    """Verify checkpoint save/load/resume mechanics."""

    def test_save_and_load_checkpoint(self, tmp_path: Path):
        """Saved checkpoint restores identical model weights."""
        config = MiniR1Config(vocab_size=100, d_model=32, n_layers=1, n_heads=2, d_ff=64, max_seq_len=16)
        model1 = MiniR1Transformer(config)
        optimizer1 = create_adamw_optimizer(model1, lr=1e-3)

        ckpt_file = tmp_path / "test_ckpt.pt"
        save_checkpoint(
            checkpoint_path=ckpt_file,
            model=model1,
            optimizer=optimizer1,
            epoch=2,
            global_step=50,
            val_loss=1.234,
            config=config,
        )

        assert ckpt_file.exists()

        # Create fresh model and restore state
        model2 = MiniR1Transformer(config)
        optimizer2 = create_adamw_optimizer(model2, lr=1e-3)

        meta = load_checkpoint(ckpt_file, model=model2, optimizer=optimizer2)

        assert meta["epoch"] == 2
        assert meta["global_step"] == 50
        assert pytest.approx(meta["val_loss"], 1e-3) == 1.234

        # Assert parameters match exactly
        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2), f"Mismatch in parameter {n1}"

    def test_resume_training_helper(self, tmp_path: Path):
        """resume_training returns epoch+1, global_step, and best_val_loss."""
        config = MiniR1Config(vocab_size=100, d_model=32, n_layers=1, n_heads=2, d_ff=64, max_seq_len=16)
        model = MiniR1Transformer(config)
        optimizer = create_adamw_optimizer(model)

        ckpt_file = tmp_path / "resume_ckpt.pt"
        save_checkpoint(ckpt_file, model, optimizer, epoch=3, global_step=120, val_loss=0.85)

        start_epoch, step, best_loss = resume_training(ckpt_file, model, optimizer)
        assert start_epoch == 4
        assert step == 120
        assert pytest.approx(best_loss, 1e-3) == 0.85
