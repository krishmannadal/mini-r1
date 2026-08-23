"""
Mini-R1 -- Checkpointing Module
===============================
Provides robust, atomic serialization and restoration of model weights, optimizer,
scheduler, AMP scaler, training counters, and RNG states.

Guarantees:
    1. Atomic writes via temporary files + os.replace (zero corruption on interruption).
    2. Rotation of periodic step checkpoints (`step_<N>.pt`) respecting retention limits.
    3. Maintenance of `latest.pt` and `best.pt`.
    4. Deterministic restoration of Python, NumPy, PyTorch CPU, and CUDA RNG states.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import glob
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.model import MiniR1Config


def _get_rng_states() -> dict[str, Any]:
    """Capture full random number generator states across all libraries."""
    states = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda_rng_state"] = torch.cuda.get_rng_state()
    return states


def _restore_rng_states(states: dict[str, Any]) -> None:
    """Restore all random number generator states."""
    if "python_rng_state" in states and states["python_rng_state"] is not None:
        random.setstate(states["python_rng_state"])

    if "numpy_rng_state" in states and states["numpy_rng_state"] is not None:
        np.random.set_state(states["numpy_rng_state"])

    if "torch_rng_state" in states and states["torch_rng_state"] is not None:
        torch.set_rng_state(states["torch_rng_state"].cpu())

    if torch.cuda.is_available() and "cuda_rng_state" in states and states["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state(states["cuda_rng_state"].cpu())


def save_checkpoint(
    checkpoint_path: Path | str,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int = 0,
    global_step: int = 0,
    val_loss: float = float("inf"),
    current_loss: float = float("inf"),
    config: MiniR1Config | dict | None = None,
    training_metrics: dict | None = None,
    dataset_state: dict | None = None,
    tokens_seen: int = 0,
) -> Path:
    """
    Atomically serialize model weights, optimizer, scheduler, scaler, and metadata to disk.
    """
    path = Path(checkpoint_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Compute parameter count
    model_params = sum(p.numel() for p in model.parameters())

    checkpoint_payload: dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": val_loss,
        "current_loss": current_loss,
        "tokens_seen": tokens_seen,
        "model_params": model_params,
        "model_state_dict": model.state_dict(),
        **_get_rng_states(),
    }

    if dataset_state is not None:
        checkpoint_payload["dataset_state"] = dataset_state

    if optimizer is not None:
        checkpoint_payload["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None and hasattr(scheduler, "state_dict"):
        checkpoint_payload["scheduler_state_dict"] = scheduler.state_dict()

    if scaler is not None and hasattr(scaler, "state_dict"):
        checkpoint_payload["scaler_state_dict"] = scaler.state_dict()

    if config is not None:
        if is_dataclass(config):
            checkpoint_payload["config"] = asdict(config)
        elif hasattr(config, "to_dict"):
            checkpoint_payload["config"] = config.to_dict()
        elif isinstance(config, dict):
            checkpoint_payload["config"] = config

    if training_metrics is not None:
        checkpoint_payload["training_metrics"] = training_metrics

    # Atomic write: write to unique temporary file in the same directory, then atomically replace target
    import uuid
    import time

    temp_path = path.parent / f"{path.stem}_{uuid.uuid4().hex}.tmp"

    try:
        with open(temp_path, "wb") as f:
            torch.save(checkpoint_payload, f)
            f.flush()
            os.fsync(f.fileno())

        # Atomically replace target file (with short retry loop for Windows filesystem locking)
        for attempt in range(5):
            try:
                os.replace(str(temp_path), str(path))
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise

    return path


def load_checkpoint(
    checkpoint_path: Path | str,
    model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    device: str = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """
    Load checkpoint and restore model, optimizer, scheduler, scaler, and RNG states.
    """
    path = Path(checkpoint_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    checkpoint = torch.load(str(path), map_location=device, weights_only=False)

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        if hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        if hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if restore_rng:
        _restore_rng_states(checkpoint)

    return checkpoint


def cleanup_stale_temp_files(checkpoint_dir: Path | str) -> list[Path]:
    """
    Remove any stale temporary checkpoint files (*.tmp, *_tmp_*.pt, etc.)
    left over from interrupted saves or crashes.
    """
    directory = Path(checkpoint_dir).resolve()
    if not directory.exists():
        return []

    cleaned: list[Path] = []
    # Patterns for temporary files
    temp_patterns = ["*.tmp", "*_tmp_*.pt", "*_tmp_*"]
    for pattern in temp_patterns:
        for p in directory.glob(pattern):
            if p.is_file():
                try:
                    p.unlink()
                    cleaned.append(p)
                except Exception:
                    pass
    return cleaned


class CheckpointManager:
    """
    Manages atomic checkpoint saving, rotation of periodic step checkpoints,
    and retention of latest.pt and best.pt.
    """

    def __init__(
        self,
        checkpoint_dir: Path | str = "checkpoints",
        retained_checkpoints: int = 3,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.retained_checkpoints = max(1, retained_checkpoints)

        # Automatically clean up any stale temporary files from previous runs
        cleanup_stale_temp_files(self.checkpoint_dir)

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        epoch: int = 0,
        global_step: int = 0,
        val_loss: float = float("inf"),
        current_loss: float = float("inf"),
        config: MiniR1Config | dict | None = None,
        is_best: bool = False,
        is_step_checkpoint: bool = True,
        training_metrics: dict | None = None,
        dataset_state: dict | None = None,
        tokens_seen: int = 0,
    ) -> dict[str, Path]:
        """
        Save latest.pt, step_<N>.pt, and optionally best.pt.
        Enforces retention limit on step_<N>.pt files.
        Optimized to write the serialized checkpoint once and copy atomically.
        """
        import shutil
        import uuid
        import time

        saved_paths: dict[str, Path] = {}

        # 1. Primary atomic save to latest.pt
        latest_path = self.checkpoint_dir / "latest.pt"
        save_checkpoint(
            checkpoint_path=latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            val_loss=val_loss,
            current_loss=current_loss,
            config=config,
            training_metrics=training_metrics,
            dataset_state=dataset_state,
            tokens_seen=tokens_seen,
        )
        saved_paths["latest"] = latest_path

        def _atomic_copy(src: Path, dst: Path) -> None:
            """Atomically copy an existing checkpoint to a target location."""
            temp_copy = dst.parent / f"{dst.stem}_{uuid.uuid4().hex}.tmp"
            try:
                shutil.copy2(str(src), str(temp_copy))
                for attempt in range(5):
                    try:
                        os.replace(str(temp_copy), str(dst))
                        break
                    except PermissionError:
                        if attempt == 4:
                            raise
                        time.sleep(0.05)
            except Exception:
                if temp_copy.exists():
                    try:
                        temp_copy.unlink()
                    except Exception:
                        pass
                raise

        # 2. Update best.pt if this is a new best
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            _atomic_copy(latest_path, best_path)
            saved_paths["best"] = best_path

        # 3. Save periodic step_<N>.pt if requested
        if is_step_checkpoint and global_step > 0:
            step_path = self.checkpoint_dir / f"step_{global_step}.pt"
            _atomic_copy(latest_path, step_path)
            saved_paths["step"] = step_path

            # Enforce retention limit
            self._purge_old_step_checkpoints()

        return saved_paths

    def _purge_old_step_checkpoints(self) -> list[Path]:
        """Remove oldest step_<N>.pt files exceeding retained_checkpoints count."""
        step_files: list[tuple[int, Path]] = []
        pattern = re.compile(r"^step_(\d+)\.pt$")

        for f in self.checkpoint_dir.glob("step_*.pt"):
            match = pattern.match(f.name)
            if match:
                step_num = int(match.group(1))
                step_files.append((step_num, f))

        step_files.sort(key=lambda x: x[0])
        purged = []

        if len(step_files) > self.retained_checkpoints:
            num_to_delete = len(step_files) - self.retained_checkpoints
            for _, f_path in step_files[:num_to_delete]:
                if f_path.exists():
                    f_path.unlink()
                    purged.append(f_path)

        return purged


def resume_training(
    checkpoint_path: Path | str,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    device: str = "cpu",
) -> tuple[int, int, float]:
    """
    Restore full training state and return (start_epoch, global_step, best_val_loss).
    """
    ckpt = load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
    )
    start_epoch = ckpt.get("epoch", 0) + 1
    global_step = ckpt.get("global_step", 0)
    best_val_loss = ckpt.get("val_loss", float("inf"))
    return start_epoch, global_step, best_val_loss
