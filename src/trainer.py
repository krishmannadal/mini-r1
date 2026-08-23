"""
Mini-R1 -- Scaled Trainer Module
================================
Coordinates mixed precision training (FP32/FP16/BF16), gradient accumulation,
gradient clipping, cosine LR scheduling, step-based validation, structured logging,
and atomic checkpointing for the 42.1M parameter SLM on RTX 4050.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable
import yaml
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.loss import compute_causal_lm_loss
from src.metrics import MetricsTracker, MetricsSummary
from src.checkpoint import save_checkpoint, load_checkpoint, CheckpointManager, resume_training, cleanup_stale_temp_files
from src.model import MiniR1Config


class Trainer:
    """
    Orchestrates scaled Transformer model training on GPU/CPU.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        config: MiniR1Config | None = None,
        scheduler: Any | None = None,
        device: str | None = None,
        amp_dtype: str = "float16",  # "float16", "bfloat16", or "float32"
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        checkpoint_dir: Path | str = "checkpoints",
        log_dir: Path | str = "logs",
        run_id: str | None = None,
        retained_checkpoints: int = 3,
        val_interval_steps: int = 20,
        checkpoint_interval_steps: int = 20,
        log_interval_steps: int = 5,
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.scheduler = scheduler

        # Detect device from model parameters if not provided
        if device is None:
            try:
                self.device = str(next(model.parameters()).device)
            except StopIteration:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.amp_dtype_str = amp_dtype.lower()
        self.grad_accum_steps = max(1, gradient_accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.val_interval_steps = val_interval_steps
        self.checkpoint_interval_steps = checkpoint_interval_steps
        self.log_interval_steps = log_interval_steps
        self.log_callback = log_callback or print

        # Setup unique run isolation
        import uuid
        self.run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Setup checkpoint directory & manager
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_mgr = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            retained_checkpoints=retained_checkpoints,
        )

        # Setup isolated run directory & logging paths
        self.base_log_dir = Path(log_dir).resolve()
        self.run_dir = self.base_log_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.run_dir / "train_log.jsonl"

        # Record system & config metadata for the run
        self._record_run_metadata()

        # Setup AMP
        self._setup_amp()

        # Training progress state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.skipped_optimizer_steps = 0
        self.tokens_seen = 0
        self.start_time = time.time()

    def _setup_amp(self) -> None:
        """Initialize AMP precision dtype and GradScaler based on device and config."""
        is_cuda = self.device.startswith("cuda") and torch.cuda.is_available()

        if not is_cuda or self.amp_dtype_str in ("float32", "fp32"):
            self.amp_enabled = False
            self.amp_dtype = torch.float32
            self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        elif self.amp_dtype_str in ("float16", "fp16"):
            self.amp_enabled = True
            self.amp_dtype = torch.float16
            self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        elif self.amp_dtype_str in ("bfloat16", "bf16"):
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
            # BF16 does not require loss scaling
            self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        else:
            raise ValueError(f"Unsupported amp_dtype: {self.amp_dtype_str}. Use float16, bfloat16, or float32.")

    def _record_run_metadata(self) -> None:
        """Record config snapshot, system information, and git commit hash in run directory."""
        # 1. Config snapshot
        config_dict: dict[str, Any] = {}
        if is_dataclass(self.config):
            config_dict["model"] = asdict(self.config)
        elif hasattr(self.config, "to_dict"):
            config_dict["model"] = self.config.to_dict()
        elif isinstance(self.config, dict):
            config_dict["model"] = self.config

        config_dict["training"] = {
            "device": self.device,
            "amp_dtype": self.amp_dtype_str,
            "gradient_accumulation_steps": self.grad_accum_steps,
            "max_grad_norm": self.max_grad_norm,
            "val_interval_steps": self.val_interval_steps,
            "checkpoint_interval_steps": self.checkpoint_interval_steps,
            "log_interval_steps": self.log_interval_steps,
        }

        with open(self.run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        # 2. System info
        is_cuda = torch.cuda.is_available()
        system_info = {
            "run_id": self.run_id,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": is_cuda,
            "cuda_version": torch.version.cuda if is_cuda else None,
            "device_name": torch.cuda.get_device_name(0) if is_cuda else "CPU",
            "gpu_vram_total_mb": (torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)) if is_cuda else 0.0,
        }
        with open(self.run_dir / "system_info.json", "w", encoding="utf-8") as f:
            json.dump(system_info, f, indent=2)

        # 3. Git commit hash
        try:
            git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=2).decode().strip()
        except Exception:
            git_hash = "unknown"
        (self.run_dir / "git_hash.txt").write_text(git_hash + "\n", encoding="utf-8")

    def log(self, message: str) -> None:
        """Output logging message."""
        self.log_callback(message)

    def _write_structured_log(self, record: dict) -> None:
        """Append a JSON line to the structured training log."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _get_gpu_memory_mb(self) -> tuple[float, float, float]:
        """Return (allocated_mb, reserved_mb, peak_allocated_mb) for current CUDA device."""
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            return allocated, reserved, peak
        return 0.0, 0.0, 0.0

    def _get_scaler_scale(self) -> float:
        """Return current scale factor from AMP GradScaler."""
        if hasattr(self.scaler, "get_scale") and hasattr(self.scaler, "is_enabled") and self.scaler.is_enabled():
            return float(self.scaler.get_scale())
        return 1.0

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> MetricsSummary:
        """
        Evaluate the model on the validation dataset under torch.no_grad() and autocast.
        """
        self.model.eval()
        tracker = MetricsTracker()
        device_type = "cuda" if self.device.startswith("cuda") else "cpu"

        for batch in dataloader:
            input_ids = batch.input_ids.to(self.device)
            targets = batch.targets.to(self.device)
            num_tokens = (targets != -100).sum().item()

            with torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                logits = self.model(input_ids)
                loss = compute_causal_lm_loss(logits, targets)

            tracker.update(loss=loss.item(), num_tokens=num_tokens)

        return tracker.compute()

    def train_epoch(
        self,
        epoch: int,
        dataloader: DataLoader,
        log_interval: int = 10,
    ) -> MetricsSummary:
        """
        Execute one complete training epoch over the dataloader (used for epoch-level runs/tests).
        """
        self.model.train()
        tracker = MetricsTracker()
        device_type = "cuda" if self.device.startswith("cuda") else "cpu"

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch.input_ids.to(self.device)
            targets = batch.targets.to(self.device)
            num_tokens = (targets != -100).sum().item()

            self.optimizer.zero_grad()
            with torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                logits = self.model(input_ids)
                loss = compute_causal_lm_loss(logits, targets)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm).item()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            tracker.update(loss=loss.item(), num_tokens=num_tokens)
            self.global_step += 1

            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(dataloader):
                curr = tracker.compute()
                self.log(
                    f"  Epoch [{epoch:>2}] | Batch [{batch_idx + 1:>3}/{len(dataloader):<3}] "
                    f"| Train Loss: {loss.item():.4f} "
                    f"| Throughput: {curr.tokens_per_sec:>7.1f} tok/s "
                    f"| Grad Norm: {grad_norm:.3f}"
                )

        return tracker.compute()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        max_steps: int | None = None,
        target_tokens: int | None = None,
        epochs: int = 1,
        resume_from: Path | str | None = None,
    ) -> list[dict]:
        """
        Execute full training with gradient accumulation, mixed precision, and checkpointing.
        Supports step-budget and token-budget termination, plus exact dataset cursor resumption.
        """
        micro_batch_size = train_loader.batch_size or 1
        effective_batch_size = micro_batch_size * self.grad_accum_steps

        if resume_from is not None:
            self.log(f"Resuming training from checkpoint: {resume_from}")
            ckpt = load_checkpoint(
                checkpoint_path=resume_from,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                device=self.device,
            )
            self.current_epoch = ckpt.get("epoch", 0) + 1
            self.global_step = ckpt.get("global_step", 0)
            self.best_val_loss = ckpt.get("val_loss", float("inf"))
            self.tokens_seen = ckpt.get("tokens_seen", self.global_step * effective_batch_size * 512)

            # Restore dataset cursor if dataset supports it
            dataset_state = ckpt.get("dataset_state")
            if dataset_state is not None and hasattr(train_loader.dataset, "load_state"):
                train_loader.dataset.load_state(dataset_state)
                self.log(
                    f"  Restored dataset position: Shard {dataset_state.get('current_shard_idx', 0)}, "
                    f"Sequence {dataset_state.get('current_seq_in_shard', 0)}, "
                    f"Tokens Seen {self.tokens_seen:,}"
                )

            self.log(
                f"Resumed successfully at Epoch {self.current_epoch}, "
                f"Global Step {self.global_step}, Best Val Loss {self.best_val_loss:.4f}\n"
            )

        history: list[dict] = []
        device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        self.start_time = time.time()

        self.log(f"+===========================================================================+")
        self.log(f"|            Mini-R1 Scaled SLM Training Loop (42.1M Parameters)            |")
        self.log(f"+===========================================================================+")
        self.log(f"  Run ID              : {self.run_id}")
        self.log(f"  Target Device       : {self.device}")
        self.log(f"  Precision Mode      : {self.amp_dtype_str.upper()} (AMP Enabled: {self.amp_enabled})")
        self.log(f"  Micro Batch Size    : {micro_batch_size}")
        self.log(f"  Accumulation Steps  : {self.grad_accum_steps}")
        self.log(f"  Effective Batch Size: {effective_batch_size}")
        self.log(f"  Grad Clipping Norm  : {self.max_grad_norm}")
        if target_tokens is not None:
            self.log(f"  Token Budget Target : {target_tokens:,} tokens")
        if max_steps is not None:
            self.log(f"  Step Budget Target  : {max_steps:,} optimizer steps")
        self.log(f"  Checkpoint Directory: {self.checkpoint_mgr.checkpoint_dir.resolve()}")
        self.log(f"  Run Log Directory   : {self.run_dir.resolve()}")
        self.log(f"  Structured Log Path : {self.log_file.resolve()}\n")

        self.model.train()
        self.optimizer.zero_grad()

        accum_loss = 0.0
        accum_tokens = 0
        micro_step = 0

        done = False
        start_epoch_loop = self.current_epoch if self.current_epoch > 0 else 1

        # Track per-step timing and reset GPU peak stats before step
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        step_start_time = time.perf_counter()

        milestones = [50_000_000, 100_000_000, 200_000_000, 300_000_000, 400_000_000, 500_000_000]
        reached_milestones = set(m for m in milestones if self.tokens_seen >= m)

        try:
            for epoch in range(start_epoch_loop, epochs + 1):
                if done:
                    break
                self.current_epoch = epoch

                for batch_idx, batch in enumerate(train_loader):
                    if max_steps is not None and self.global_step >= max_steps:
                        done = True
                        break
                    if target_tokens is not None and self.tokens_seen >= target_tokens:
                        done = True
                        break

                    input_ids = batch.input_ids.to(self.device)
                    targets = batch.targets.to(self.device)
                    num_tokens = (targets != -100).sum().item()
                    self.tokens_seen += num_tokens

                    # Forward pass with AMP autocast
                    with torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                        logits = self.model(input_ids)
                        loss = compute_causal_lm_loss(logits, targets)
                        # Scale loss for gradient accumulation
                        scaled_loss = loss / self.grad_accum_steps

                    # Backward pass with GradScaler
                    self.scaler.scale(scaled_loss).backward()

                    accum_loss += loss.item()
                    accum_tokens += num_tokens
                    micro_step += 1

                    # Check if accumulation window is complete
                    if micro_step % self.grad_accum_steps == 0:
                        # Unscale gradients before clipping
                        self.scaler.unscale_(self.optimizer)

                        # Gradient clipping
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm,
                        ).item()

                        # Optimizer step through scaler
                        scale_before = self._get_scaler_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self._get_scaler_scale()
                        if scale_after < scale_before:
                            self.skipped_optimizer_steps += 1
                        self.optimizer.zero_grad()

                        # Step LR scheduler
                        current_lr = self.optimizer.param_groups[0]["lr"]
                        if self.scheduler is not None:
                            self.scheduler.step()

                        self.global_step += 1

                        # Compute true per-step throughput (wall-clock of this step only)
                        step_duration = max(1e-6, time.perf_counter() - step_start_time)
                        tokens_per_sec_step = accum_tokens / step_duration

                        avg_step_loss = accum_loss / self.grad_accum_steps
                        alloc_mb, res_mb, peak_mb = self._get_gpu_memory_mb()
                        scaler_scale = self._get_scaler_scale()
                        elapsed = time.time() - self.start_time

                        # Check milestone crossings
                        for m in milestones:
                            if self.tokens_seen >= m and m not in reached_milestones:
                                reached_milestones.add(m)
                                self.log(
                                    f"\n  +===========================================================================+\n"
                                    f"  |  >>> [MILESTONE REACHED] {m/1e6:.0f}M TOKENS PROCESSED (Step {self.global_step}) | Loss: {avg_step_loss:.4f}\n"
                                    f"  +===========================================================================+\n"
                                )

                        # 1. Periodic Logging
                        token_info = ""
                        pct = None
                        if target_tokens is not None:
                            pct = (self.tokens_seen / target_tokens) * 100.0
                            token_info = f"| Tokens: {self.tokens_seen/1e6:.2f}M/{target_tokens/1e6:.0f}M ({pct:.1f}%) "
                        else:
                            token_info = f"| Tokens: {self.tokens_seen/1e6:.2f}M "

                        if self.global_step % self.log_interval_steps == 0 or (max_steps and self.global_step >= max_steps):
                            self.log(
                                f"  Step [{self.global_step:>5}] | Epoch [{epoch:>2}] "
                                f"{token_info}"
                                f"| Loss: {avg_step_loss:.4f} "
                                f"| LR: {current_lr:.2e} "
                                f"| GradNorm: {grad_norm:.3f} "
                                f"| Tok/s: {tokens_per_sec_step:>6.1f} "
                                f"| Scale: {scaler_scale:.0f} "
                                f"| VRAM: {alloc_mb:.0f}/{peak_mb:.0f}/{res_mb:.0f} MB"
                            )

                            log_entry = {
                                "run_id": self.run_id,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "global_step": self.global_step,
                                "epoch": epoch,
                                "train_loss": avg_step_loss,
                                "lr": current_lr,
                                "grad_norm": grad_norm,
                                "tokens_per_sec_step": tokens_per_sec_step,
                                "tokens_seen": self.tokens_seen,
                                "percent_complete": pct,
                                "scaler_scale": scaler_scale,
                                "skipped_optimizer_steps": self.skipped_optimizer_steps,
                                "gpu_allocated_mb": alloc_mb,
                                "gpu_reserved_mb": res_mb,
                                "gpu_peak_allocated_mb": peak_mb,
                                "elapsed_sec": elapsed,
                            }
                            self._write_structured_log(log_entry)

                        # 2. Periodic Validation
                        val_loss = None
                        is_new_best = False
                        if val_loader is not None and (
                            self.global_step % self.val_interval_steps == 0
                            or (max_steps and self.global_step >= max_steps)
                        ):
                            val_summary = self.evaluate(val_loader)
                            val_loss = val_summary.avg_loss
                            is_new_best = val_loss < self.best_val_loss

                            if is_new_best:
                                self.best_val_loss = val_loss

                            self.log(
                                f"  >> [VAL] Step [{self.global_step:>4}] | Val Loss: {val_loss:.4f} "
                                f"| Perplexity: {val_summary.perplexity:.2f} "
                                f"| Best Val: {self.best_val_loss:.4f}"
                            )

                            # Write structured val record
                            self._write_structured_log({
                                "run_id": self.run_id,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "global_step": self.global_step,
                                "epoch": epoch,
                                "val_loss": val_loss,
                                "val_perplexity": val_summary.perplexity,
                                "is_best": is_new_best,
                                "lr": current_lr,
                                "grad_norm": grad_norm,
                                "tokens_per_sec_step": tokens_per_sec_step,
                                "tokens_seen": self.tokens_seen,
                                "scaler_scale": scaler_scale,
                                "skipped_optimizer_steps": self.skipped_optimizer_steps,
                                "gpu_allocated_mb": alloc_mb,
                                "gpu_reserved_mb": res_mb,
                                "gpu_peak_allocated_mb": peak_mb,
                                "elapsed_sec": elapsed,
                            })
                            self.model.train()

                        # 3. Consolidated Checkpointing (no duplicate writes)
                        is_step_ckpt = (
                            self.global_step % self.checkpoint_interval_steps == 0
                            or (max_steps and self.global_step >= max_steps)
                        )
                        if is_new_best or is_step_ckpt:
                            ds_state = None
                            if hasattr(train_loader.dataset, "get_state"):
                                ds_state = train_loader.dataset.get_state()

                            saved = self.checkpoint_mgr.save(
                                model=self.model,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                scaler=self.scaler,
                                epoch=epoch,
                                global_step=self.global_step,
                                val_loss=val_loss if val_loss is not None else self.best_val_loss,
                                current_loss=avg_step_loss,
                                config=self.config,
                                is_best=is_new_best,
                                is_step_checkpoint=is_step_ckpt,
                                training_metrics={
                                    "train_loss": avg_step_loss,
                                    "grad_norm": grad_norm,
                                    "tokens_per_sec": tokens_per_sec_step,
                                    "tokens_seen": self.tokens_seen,
                                },
                                dataset_state=ds_state,
                                tokens_seen=self.tokens_seen,
                            )
                            saved_names = [p.name for p in saved.values()]
                            self.log(f"  >> [CHECKPOINT] Saved: {', '.join(saved_names)}")

                        # Reset accumulation buffers and restart step timer / peak memory
                        accum_loss = 0.0
                        accum_tokens = 0
                        if torch.cuda.is_available() and self.device.startswith("cuda"):
                            torch.cuda.reset_peak_memory_stats()
                        step_start_time = time.perf_counter()

                # Flush remaining accumulated gradients if any at epoch end
                if micro_step % self.grad_accum_steps != 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm).item()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.global_step += 1

                    step_duration = max(1e-6, time.perf_counter() - step_start_time)
                    tokens_per_sec_step = accum_tokens / step_duration
                    avg_step_loss = accum_loss / (micro_step % self.grad_accum_steps)
                    alloc_mb, res_mb, peak_mb = self._get_gpu_memory_mb()
                    scaler_scale = self._get_scaler_scale()
                    elapsed = time.time() - self.start_time

                    # Structured log for flushed step
                    log_entry = {
                        "run_id": self.run_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "global_step": self.global_step,
                        "epoch": epoch,
                        "train_loss": avg_step_loss,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "grad_norm": grad_norm,
                        "tokens_per_sec_step": tokens_per_sec_step,
                        "tokens_seen": self.tokens_seen,
                        "scaler_scale": scaler_scale,
                        "gpu_allocated_mb": alloc_mb,
                        "gpu_reserved_mb": res_mb,
                        "gpu_peak_allocated_mb": peak_mb,
                        "elapsed_sec": elapsed,
                    }
                    self._write_structured_log(log_entry)

                    accum_loss = 0.0
                    accum_tokens = 0
                    micro_step = 0

                # Epoch-end validation and checkpointing
                val_loss = None
                is_new_best = False
                if val_loader is not None:
                    val_summary = self.evaluate(val_loader)
                    val_loss = val_summary.avg_loss
                    is_new_best = val_loss < self.best_val_loss
                    if is_new_best:
                        self.best_val_loss = val_loss

                    self._write_structured_log({
                        "run_id": self.run_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "global_step": self.global_step,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_perplexity": val_summary.perplexity,
                        "is_best": is_new_best,
                        "lr": self.optimizer.param_groups[0]["lr"] if self.optimizer is not None else 0.0,
                        "grad_norm": grad_norm if 'grad_norm' in locals() else 0.0,
                        "tokens_per_sec_step": tokens_per_sec_step if 'tokens_per_sec_step' in locals() else 0.0,
                        "tokens_seen": self.tokens_seen,
                        "scaler_scale": scaler_scale if 'scaler_scale' in locals() else 1.0,
                        "gpu_allocated_mb": alloc_mb if 'alloc_mb' in locals() else 0.0,
                        "gpu_reserved_mb": res_mb if 'res_mb' in locals() else 0.0,
                        "gpu_peak_allocated_mb": peak_mb if 'peak_mb' in locals() else 0.0,
                        "elapsed_sec": time.time() - self.start_time,
                    })

                # Save latest.pt (and best.pt if is_new_best) at epoch boundary
                ds_state = None
                if hasattr(train_loader.dataset, "get_state"):
                    ds_state = train_loader.dataset.get_state()

                self.checkpoint_mgr.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch,
                    global_step=self.global_step,
                    val_loss=val_loss if val_loss is not None else self.best_val_loss,
                    current_loss=avg_step_loss if 'avg_step_loss' in locals() else 0.0,
                    config=self.config,
                    is_best=is_new_best,
                    is_step_checkpoint=False,
                    dataset_state=ds_state,
                    tokens_seen=self.tokens_seen,
                )

                # Record epoch history
                history.append({
                    "epoch": epoch,
                    "train_loss": avg_step_loss if 'avg_step_loss' in locals() else 0.0,
                    "val_loss": val_loss if val_loss is not None else self.best_val_loss,
                    "global_step": self.global_step,
                })

        except Exception as e:
            self.log(f"\n[FATAL ERROR] Training interrupted by exception: {e}")
            try:
                ds_state = None
                if hasattr(train_loader.dataset, "get_state"):
                    ds_state = train_loader.dataset.get_state()
                rec_path = self.checkpoint_dir / "emergency_recovery.pt"
                save_checkpoint(
                    checkpoint_path=rec_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=self.current_epoch,
                    global_step=self.global_step,
                    val_loss=self.best_val_loss,
                    config=self.config,
                    dataset_state=ds_state,
                    tokens_seen=self.tokens_seen,
                )
                self.log(f"[RECOVERY] Saved emergency checkpoint: {rec_path}")
            except Exception as save_err:
                self.log(f"[RECOVERY FAILED] Could not save emergency checkpoint: {save_err}")
            raise e

        self.log(f"\n[OK] Training completed at Global Step {self.global_step} (Tokens Seen: {self.tokens_seen:,}).")
        return history
