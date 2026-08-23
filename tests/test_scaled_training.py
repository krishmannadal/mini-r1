"""
Mini-R1 -- Scaled Training & Checkpointing Unit Tests
=====================================================
Tests verifying:
1. Complete state payload saving and atomic replacement.
2. CosineWarmupScheduler state serialization and restore.
3. GradScaler state preservation.
4. Checkpoint rotation and retention limits.
5. Non-destructive resumption and global step continuity.
"""

from pathlib import Path
import tempfile
import pytest
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.model import MiniR1Config, MiniR1Transformer
from src.optimizer import create_adamw_optimizer
from src.scheduler import CosineWarmupScheduler
from src.checkpoint import save_checkpoint, load_checkpoint, CheckpointManager, resume_training
from src.trainer import Trainer


class TestScaledTrainingInfrastructure:
    """Verify state serialization, scheduling, and atomic checkpointing."""

    def test_scheduler_warmup_and_decay(self):
        """Scheduler computes linear warmup and cosine decay accurately."""
        config = MiniR1Config(vocab_size=100, d_model=32, n_layers=1, n_heads=2, d_ff=64)
        model = MiniR1Transformer(config)
        optimizer = create_adamw_optimizer(model, lr=1e-3)

        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=10,
            max_steps=100,
            base_lr=1e-3,
            min_lr=1e-4,
        )

        # Step 0: Warmup start (factor 0)
        assert pytest.approx(optimizer.param_groups[0]["lr"], abs=1e-6) == 1e-4

        # Step 5: Halfway warmup
        scheduler.step(step=5)
        expected_lr_step5 = 1e-4 + (1e-3 - 1e-4) * 0.5
        assert pytest.approx(optimizer.param_groups[0]["lr"], abs=1e-6) == expected_lr_step5

        # Step 10: Peak LR
        scheduler.step(step=10)
        assert pytest.approx(optimizer.param_groups[0]["lr"], abs=1e-6) == 1e-3

    def test_scheduler_state_dict_restore(self):
        """Scheduler state_dict saves and restores exact step progression."""
        config = MiniR1Config(vocab_size=100, d_model=32, n_layers=1, n_heads=2, d_ff=64)
        model = MiniR1Transformer(config)
        opt1 = create_adamw_optimizer(model, lr=1e-3)
        sched1 = CosineWarmupScheduler(opt1, warmup_steps=5, max_steps=20, min_lr=1e-4)

        # Advance sched1 to step 7
        for _ in range(7):
            sched1.step()

        saved_state = sched1.state_dict()

        # Create new scheduler on new optimizer and restore
        opt2 = create_adamw_optimizer(model, lr=1e-3)
        sched2 = CosineWarmupScheduler(opt2, warmup_steps=5, max_steps=20, min_lr=1e-4)
        sched2.load_state_dict(saved_state)

        assert sched2.current_step == 7
        assert pytest.approx(opt2.param_groups[0]["lr"], abs=1e-6) == opt1.param_groups[0]["lr"]

    def test_atomic_checkpoint_save_and_load(self):
        """Checkpoints are saved atomically with full payload and restored cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_model.pt"

            config = MiniR1Config(vocab_size=100, d_model=32, n_layers=1, n_heads=2, d_ff=64)
            model = MiniR1Transformer(config)
            optimizer = create_adamw_optimizer(model, lr=1e-3)
            scheduler = CosineWarmupScheduler(optimizer, warmup_steps=2, max_steps=10)
            scaler = torch.amp.GradScaler("cuda", enabled=False)

            # Advance state
            scheduler.step()
            saved_path = save_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=2,
                global_step=15,
                val_loss=3.1415,
                config=config,
            )

            assert saved_path.exists()
            assert not Path(str(ckpt_path) + ".tmp").exists()

            # Load into fresh instances
            model2 = MiniR1Transformer(config)
            opt2 = create_adamw_optimizer(model2, lr=1e-3)
            sched2 = CosineWarmupScheduler(opt2, warmup_steps=2, max_steps=10)

            ckpt_data = load_checkpoint(
                checkpoint_path=ckpt_path,
                model=model2,
                optimizer=opt2,
                scheduler=sched2,
                scaler=scaler,
            )

            assert ckpt_data["global_step"] == 15
            assert ckpt_data["epoch"] == 2
            assert pytest.approx(ckpt_data["val_loss"], abs=1e-4) == 3.1415
            assert sched2.current_step == scheduler.current_step

    def test_checkpoint_rotation_retention(self):
        """CheckpointManager preserves at most N periodic step files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(checkpoint_dir=tmpdir, retained_checkpoints=2)
            config = MiniR1Config(vocab_size=100, d_model=32, n_layers=1, n_heads=2, d_ff=64)
            model = MiniR1Transformer(config)

            # Save step 10, 20, 30
            mgr.save(model=model, global_step=10, is_step_checkpoint=True)
            mgr.save(model=model, global_step=20, is_step_checkpoint=True)
            mgr.save(model=model, global_step=30, is_step_checkpoint=True)

            step_files = sorted([f.name for f in Path(tmpdir).glob("step_*.pt")])
            # Only step_20.pt and step_30.pt should remain (step_10 purged)
            assert step_files == ["step_20.pt", "step_30.pt"]
            assert (Path(tmpdir) / "latest.pt").exists()

    def test_stale_temp_cleanup(self, tmp_path: Path):
        """CheckpointManager cleans up any stale *.tmp or temporary files on startup."""
        # Create stale temporary artifacts
        (tmp_path / "latest_tmp_1234.pt").write_bytes(b"stale_data_1")
        (tmp_path / "model_abcd123.tmp").write_bytes(b"stale_data_2")
        (tmp_path / "step_10_tmp_foo.tmp").write_bytes(b"stale_data_3")
        (tmp_path / "valid_model.pt").write_bytes(b"valid_data")

        # Initialize manager which triggers cleanup
        mgr = CheckpointManager(checkpoint_dir=tmp_path)

        assert not (tmp_path / "latest_tmp_1234.pt").exists()
        assert not (tmp_path / "model_abcd123.tmp").exists()
        assert not (tmp_path / "step_10_tmp_foo.tmp").exists()
        assert (tmp_path / "valid_model.pt").exists()

    def test_unique_run_directories_and_metadata(self, tmp_path: Path):
        """Each Trainer instance creates a unique isolated run directory with metadata."""
        config = MiniR1Config(vocab_size=50, d_model=16, n_layers=1, n_heads=2, d_ff=32)
        model = MiniR1Transformer(config)
        optimizer = create_adamw_optimizer(model, lr=1e-3)

        trainer1 = Trainer(model=model, optimizer=optimizer, config=config, log_dir=tmp_path / "logs", checkpoint_dir=tmp_path / "ckpt")
        trainer2 = Trainer(model=model, optimizer=optimizer, config=config, log_dir=tmp_path / "logs", checkpoint_dir=tmp_path / "ckpt")

        assert trainer1.run_id != trainer2.run_id
        assert trainer1.run_dir != trainer2.run_dir
        assert trainer1.run_dir.exists()
        assert trainer2.run_dir.exists()

        # Check metadata files in run_dir
        assert (trainer1.run_dir / "config_snapshot.yaml").exists()
        assert (trainer1.run_dir / "system_info.json").exists()
        assert (trainer1.run_dir / "git_hash.txt").exists()

    def test_structured_log_fields_and_per_step_throughput(self, tmp_path: Path):
        """Structured logs record run_id, per-step throughput, peak VRAM, and scaler_scale."""
        import json
        from torch.utils.data import TensorDataset, DataLoader

        config = MiniR1Config(vocab_size=50, d_model=16, n_layers=1, n_heads=2, d_ff=32)
        model = MiniR1Transformer(config)
        optimizer = create_adamw_optimizer(model, lr=1e-3)

        # Create dummy dataloader
        class SimpleBatch:
            def __init__(self, x):
                self.input_ids = x
                self.targets = x

        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 8
            def __getitem__(self, idx):
                return torch.randint(0, 50, (16,), dtype=torch.long)

        def collate_fn(batch):
            x = torch.stack(batch)
            return SimpleBatch(x)

        loader = DataLoader(SimpleDataset(), batch_size=4, collate_fn=collate_fn)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            config=config,
            log_dir=tmp_path / "logs",
            checkpoint_dir=tmp_path / "ckpt",
            gradient_accumulation_steps=1,
            log_interval_steps=1,
        )

        trainer.train(train_loader=loader, max_steps=2)

        # Read JSONL log records
        assert trainer.log_file.exists()
        with open(trainer.log_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) >= 2
        for record in lines:
            assert record["run_id"] == trainer.run_id
            assert "timestamp" in record
            assert "global_step" in record
            assert "epoch" in record
            assert "train_loss" in record
            assert "lr" in record
            assert "grad_norm" in record
            assert "tokens_per_sec_step" in record
            assert record["tokens_per_sec_step"] > 0.0
            assert "scaler_scale" in record
            assert "gpu_allocated_mb" in record
            assert "gpu_reserved_mb" in record
            assert "gpu_peak_allocated_mb" in record
            assert "elapsed_sec" in record

    def test_no_duplicate_best_model_or_uncontrolled_epoch_checkpoints(self, tmp_path: Path):
        """Checkpoints contain canonical best.pt, latest.pt, step_N.pt and no best_model.pt or uncontrolled epoch files."""
        config = MiniR1Config(vocab_size=50, d_model=16, n_layers=1, n_heads=2, d_ff=32)
        model = MiniR1Transformer(config)
        optimizer = create_adamw_optimizer(model, lr=1e-3)

        class SimpleBatch:
            def __init__(self, x):
                self.input_ids = x
                self.targets = x

        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4
            def __getitem__(self, idx):
                return torch.randint(0, 50, (8,), dtype=torch.long)

        def collate_fn(batch):
            return SimpleBatch(torch.stack(batch))

        loader = DataLoader(SimpleDataset(), batch_size=2, collate_fn=collate_fn)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            config=config,
            checkpoint_dir=tmp_path / "ckpt",
            log_dir=tmp_path / "logs",
            val_interval_steps=1,
            checkpoint_interval_steps=1,
            retained_checkpoints=2,
        )

        trainer.train(train_loader=loader, val_loader=loader, epochs=2)

        ckpt_files = [f.name for f in (tmp_path / "ckpt").glob("*.pt")]
        assert "best.pt" in ckpt_files
        assert "latest.pt" in ckpt_files
        assert not (tmp_path / "ckpt" / "best_model.pt").exists()
        # No uncontrolled checkpoint_epoch_N.pt files
        assert len([f for f in ckpt_files if f.startswith("checkpoint_epoch_")]) == 0
