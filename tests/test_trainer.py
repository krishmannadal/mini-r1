"""
Mini-R1 -- Trainer Test Suite
=============================
Tests verifying:
1. Trainer.train_epoch runs and returns valid metrics.
2. Trainer.evaluate runs and computes validation loss.
3. Multi-epoch training loop saves periodic checkpoints and best model.
4. Resuming training from a saved checkpoint continues from the correct epoch.
"""

from pathlib import Path
import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.model import MiniR1Transformer, MiniR1Config
from src.optimizer import create_adamw_optimizer
from src.dataloader import create_dataloader, CausalCollateFn
from src.trainer import Trainer


class DummyDataset(torch.utils.data.Dataset):
    """Synthetic dataset generating random sequence tensors."""
    def __init__(self, count: int = 16, seq_len: int = 12, vocab_size: int = 100):
        self.count = count
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.count

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.randint(10, self.vocab_size, (self.seq_len,), dtype=torch.long)


class TestTrainerExecution:
    """Verify Trainer loops, evaluation, and checkpoint management."""

    def test_train_epoch_and_evaluate(self, tmp_path: Path):
        """Single epoch training and evaluation return non-zero metrics."""
        config = MiniR1Config(vocab_size=100, d_model=32, n_layers=1, n_heads=2, d_ff=64, max_seq_len=16)
        model = MiniR1Transformer(config)
        optimizer = create_adamw_optimizer(model, lr=1e-3)

        ds = DummyDataset(count=8, seq_len=10, vocab_size=100)
        loader = create_dataloader(ds, batch_size=4, shuffle=False)

        trainer = Trainer(model=model, optimizer=optimizer, config=config, checkpoint_dir=tmp_path)

        train_metrics = trainer.train_epoch(epoch=1, dataloader=loader, log_interval=10)
        assert train_metrics.avg_loss > 0.0
        assert train_metrics.tokens_per_sec > 0.0

        val_metrics = trainer.evaluate(loader)
        assert val_metrics.avg_loss > 0.0

    def test_multi_epoch_training_and_checkpoint_saving(self, tmp_path: Path):
        """Full trainer run executes 2 epochs and saves checkpoints."""
        config = MiniR1Config(vocab_size=100, d_model=32, n_layers=1, n_heads=2, d_ff=64, max_seq_len=16)
        model = MiniR1Transformer(config)
        optimizer = create_adamw_optimizer(model, lr=5e-3)

        ds = DummyDataset(count=8, seq_len=10, vocab_size=100)
        loader = create_dataloader(ds, batch_size=4, shuffle=False)

        trainer = Trainer(model=model, optimizer=optimizer, config=config, checkpoint_dir=tmp_path)
        history = trainer.train(train_loader=loader, val_loader=loader, epochs=2)

        assert len(history) == 2
        assert (tmp_path / "best.pt").exists()
        assert (tmp_path / "latest.pt").exists()
        assert not (tmp_path / "best_model.pt").exists(), "best_model.pt legacy duplicate should not be saved"

    def test_resume_training_flow(self, tmp_path: Path):
        """Training can resume from a checkpoint and continue additional epochs."""
        config = MiniR1Config(vocab_size=100, d_model=32, n_layers=1, n_heads=2, d_ff=64, max_seq_len=16)
        model = MiniR1Transformer(config)
        optimizer = create_adamw_optimizer(model, lr=5e-3)

        ds = DummyDataset(count=8, seq_len=10, vocab_size=100)
        loader = create_dataloader(ds, batch_size=4, shuffle=False)

        trainer = Trainer(model=model, optimizer=optimizer, config=config, checkpoint_dir=tmp_path)
        trainer.train(train_loader=loader, val_loader=loader, epochs=1)

        latest_ckpt = tmp_path / "latest.pt"
        assert latest_ckpt.exists()

        # Resume to epoch 2
        trainer2 = Trainer(model=model, optimizer=optimizer, config=config, checkpoint_dir=tmp_path)
        history = trainer2.train(train_loader=loader, val_loader=loader, epochs=2, resume_from=latest_ckpt)

        assert len(history) == 1
        assert history[0]["epoch"] == 2
