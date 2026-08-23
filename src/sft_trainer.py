"""
Mini-R1 -- Masked SFT Trainer Module
====================================
Orchestrates Masked Supervised Fine-Tuning (SFT) for Mini-R1 reasoning experiments.

Key Features:
1. Strict initialization from immutable base checkpoint (best.pt, step 18,000, val loss 2.8726).
2. Masked causal LM next-token prediction:
   - Prompt tokens: label = -100 (ignored in loss and gradients)
   - Completion tokens: active cross-entropy loss
3. Granular token accounting:
   - Total tokens seen
   - Completion tokens seen (active loss positions)
   - Prompt tokens seen (ignored positions)
4. Dev-set evaluation with completion perplexity and format compliance checks.
5. Atomic checkpointing with RNG state capture and run isolation.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable
import yaml
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    CheckpointManager,
    _get_rng_states,
    _restore_rng_states,
)
from src.evaluation import FormatValidator, AnswerExtractor, generate_autoregressive
from src.loss import compute_causal_lm_loss
from src.model import MiniR1Transformer, MiniR1Config
from src.optimizer import create_adamw_optimizer
from src.scheduler import CosineWarmupScheduler
from src.sft_dataset import SFTDataset, collate_sft_batch
from src.tokenizer import Tokenizer


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA256 hash of a file."""
    p = Path(path)
    if not p.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class SFTTrainer:
    """
    Supervised Fine-Tuning Trainer for Mini-R1.
    """

    def __init__(
        self,
        model: MiniR1Transformer,
        tokenizer: Tokenizer,
        config: dict[str, Any],
        optimizer: Optimizer | None = None,
        scheduler: Any | None = None,
        device: str | None = None,
        base_checkpoint_path: str | Path | None = None,
        run_id: str | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.log_callback = log_callback or print

        # Device detection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)

        # Base checkpoint provenance
        self.base_checkpoint_path = Path(base_checkpoint_path) if base_checkpoint_path else None
        self.base_checkpoint_hash = (
            compute_file_sha256(self.base_checkpoint_path)
            if self.base_checkpoint_path and self.base_checkpoint_path.exists()
            else "NONE"
        )

        # Hyperparameters
        train_cfg = config.get("training", {})
        self.learning_rate = float(train_cfg.get("learning_rate", 1e-4))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.01))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        self.grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", 4))
        self.micro_batch_size = int(train_cfg.get("micro_batch_size", 8))
        self.max_steps = int(train_cfg.get("max_steps", 500))
        self.warmup_steps = int(train_cfg.get("warmup_steps", 50))
        self.min_lr = float(train_cfg.get("min_lr", 1e-5))
        self.amp_dtype_str = str(train_cfg.get("amp_dtype", "float16")).lower()
        self.val_interval_steps = int(train_cfg.get("val_interval_steps", 50))
        self.checkpoint_interval_steps = int(train_cfg.get("checkpoint_interval_steps", 50))
        self.log_interval_steps = int(train_cfg.get("log_interval_steps", 10))
        self.condition = str(config.get("experiment", {}).get("condition", "cot")).lower()
        self.seed = int(config.get("experiment", {}).get("seed", 42))

        # Setup run directories
        import uuid
        condition_dir = "sft_direct" if self.condition == "direct" else "sft_cot"
        self.run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.output_dir = Path("runs") / condition_dir / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "sft_train_log.jsonl"

        # Checkpoint manager
        self.checkpoint_mgr = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            retained_checkpoints=int(train_cfg.get("retained_checkpoints", 3)),
        )

        # Setup AMP
        self._setup_amp()

        # Setup Optimizer & Scheduler if not injected
        if optimizer is None:
            self.optimizer = create_adamw_optimizer(
                model=self.model,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.95),
                eps=1e-8,
            )
        else:
            self.optimizer = optimizer

        if scheduler is None:
            self.scheduler = CosineWarmupScheduler(
                optimizer=self.optimizer,
                warmup_steps=self.warmup_steps,
                max_steps=self.max_steps,
                base_lr=self.learning_rate,
                min_lr=self.min_lr,
            )
        else:
            self.scheduler = scheduler

        # Training state counters
        self.global_step = 0
        self.current_epoch = 0
        self.tokens_seen = 0
        self.completion_tokens_seen = 0
        self.prompt_tokens_seen = 0
        self.best_val_loss = float("inf")
        self.start_time = time.time()

        # Log startup configuration
        self._log_startup_provenance()

    def _setup_amp(self) -> None:
        """Initialize AMP precision dtype and GradScaler based on device."""
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
            self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        else:
            raise ValueError(f"Unsupported amp_dtype: {self.amp_dtype_str}")

    def _log_startup_provenance(self) -> None:
        """Log full provenance, hashes, parameter counts, and hyperparameter configuration."""
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "sft_startup",
            "run_id": self.run_id,
            "condition": self.condition,
            "device": self.device,
            "amp_dtype": self.amp_dtype_str,
            "model_parameters": num_params,
            "trainable_parameters": num_trainable,
            "base_checkpoint_path": str(self.base_checkpoint_path) if self.base_checkpoint_path else None,
            "base_checkpoint_sha256": self.base_checkpoint_hash,
            "tokenizer_path": str(self.tokenizer.model_path),
            "tokenizer_sha256": compute_file_sha256(self.tokenizer.model_path),
            "config": self.config,
        }

        # Save metadata to run directory
        meta_path = self.output_dir / "run_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        self.log_callback(
            f"=== Mini-R1 SFT Startup [{self.condition.upper()}] ===\n"
            f"Run ID:            {self.run_id}\n"
            f"Output Dir:        {self.output_dir}\n"
            f"Base Checkpoint:   {self.base_checkpoint_path} (SHA256: {self.base_checkpoint_hash[:12]}...)\n"
            f"Model Parameters:  {num_params:,} ({num_trainable:,} trainable)\n"
            f"Precision:         {self.amp_dtype_str} (Scaler enabled={self.scaler.is_enabled()})\n"
            f"Effective Batch:   {self.micro_batch_size * self.grad_accum_steps} "
            f"(micro={self.micro_batch_size}, accum={self.grad_accum_steps})\n"
            f"Peak LR:           {self.learning_rate:.2e} (min={self.min_lr:.2e}, warmup={self.warmup_steps})\n"
            f"======================================================"
        )

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
        is_accumulating: bool = False,
    ) -> tuple[float, int, int]:
        """
        Execute forward pass, loss calculation, backward pass, and token accounting for one micro-batch.

        Args:
            batch: Dictionary containing 'input_ids', 'labels', 'attention_mask',
                   'prompt_tokens_count', 'completion_tokens_count'
            is_accumulating: Whether this micro-batch is part of gradient accumulation.

        Returns:
            Tuple of (unscaled_loss_float, num_completion_tokens, num_total_tokens)
        """
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)

        batch_size, seq_len = input_ids.shape
        num_comp_tokens = int(batch["completion_tokens_count"].sum().item())
        num_total_tokens = batch_size * seq_len
        num_prompt_tokens = int(batch["prompt_tokens_count"].sum().item())

        # Forward pass under AMP autocast
        device_type = "cuda" if "cuda" in self.device else "cpu"
        with torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            logits = self.model(input_ids)
            # Masked causal next-token cross entropy loss
            loss = compute_causal_lm_loss(logits, labels, ignore_index=-100)

            # Scale loss for gradient accumulation
            scaled_loss = loss / self.grad_accum_steps

        # Backward pass with GradScaler
        self.scaler.scale(scaled_loss).backward()

        # Update running token counters
        self.tokens_seen += num_total_tokens
        self.completion_tokens_seen += num_comp_tokens
        self.prompt_tokens_seen += num_prompt_tokens

        return loss.item(), num_comp_tokens, num_total_tokens

    def optimizer_step(self) -> tuple[float, bool]:
        """
        Unscale gradients, clip norms, advance optimizer, update scheduler, and step scaler.

        Returns:
            Tuple of (grad_norm, was_step_successful)
        """
        # Unscale gradients before clipping
        self.scaler.unscale_(self.optimizer)

        # Clip gradient norm across all model parameters
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm))

        # Check for NaN / Inf
        if math.isnan(grad_norm) or math.isinf(grad_norm):
            self.log_callback(f"[WARNING] Step {self.global_step}: Non-finite grad_norm ({grad_norm}). Skipping step.")
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            return grad_norm, False

        # Step optimizer and update scaler
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        # Advance learning rate scheduler
        self.global_step += 1
        self.scheduler.step(step=self.global_step)

        return grad_norm, True

    def evaluate_dev(
        self,
        dev_loader: DataLoader,
        num_qualitative_samples: int = 5,
    ) -> dict[str, Any]:
        """
        Evaluate frozen model on dev set (masked loss, completion perplexity, format compliance).

        Does NOT touch test sets.
        """
        self.model.eval()
        total_loss_sum = 0.0
        total_comp_tokens = 0
        total_examples = 0

        start_t = time.perf_counter()

        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)
                comp_counts = batch["completion_tokens_count"].to(self.device)

                device_type = "cuda" if "cuda" in self.device else "cpu"
                with torch.amp.autocast(device_type=device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                    logits = self.model(input_ids)
                    loss = compute_causal_lm_loss(logits, labels, ignore_index=-100)

                batch_comp = int(comp_counts.sum().item())
                total_loss_sum += loss.item() * batch_comp
                total_comp_tokens += batch_comp
                total_examples += input_ids.size(0)

        elapsed = max(1e-5, time.perf_counter() - start_t)
        avg_loss = total_loss_sum / max(1, total_comp_tokens)
        ppl = math.exp(min(50.0, avg_loss))

        # Qualitative greedy evaluation on first few dev examples
        qualitative_results = []
        format_passed_count = 0
        answer_tag_present_count = 0

        # Sample up to num_qualitative_samples items from dataset
        raw_dataset: SFTDataset = dev_loader.dataset  # type: ignore
        eval_sample_indices = range(min(num_qualitative_samples, len(raw_dataset)))

        for idx in eval_sample_indices:
            ex = raw_dataset[idx]
            prompt = ex["prompt_text"]
            prompt_ids = self.tokenizer.encode(prompt, add_bos=False, add_eos=False)

            gen_out = generate_autoregressive(
                model=self.model,
                prompt_token_ids=prompt_ids,
                tokenizer=self.tokenizer,
                max_new_tokens=256,
                temperature=0.0,
                device=self.device,
                stop_token_ids=[self.tokenizer.eos_id, self.tokenizer.answer_end_id],
            )

            validation = FormatValidator.validate_formatting(gen_out.generated_text)
            has_answer = validation["has_answer_open"] and validation["has_answer_close"]
            if validation["is_fully_formatted"]:
                format_passed_count += 1
            if has_answer:
                answer_tag_present_count += 1

            qualitative_results.append({
                "id": ex.get("id", f"sample_{idx}"),
                "prompt": prompt,
                "gold_answer": ex.get("completion_text", ""),
                "generated_text": gen_out.generated_text,
                "is_fully_formatted": validation["is_fully_formatted"],
                "has_answer_tag": has_answer,
            })

        num_qual = max(1, len(eval_sample_indices))
        format_compliance_rate = format_passed_count / num_qual
        answer_tag_rate = answer_tag_present_count / num_qual

        self.model.train()

        return {
            "eval_loss": avg_loss,
            "perplexity": ppl,
            "total_examples": total_examples,
            "total_completion_tokens": total_comp_tokens,
            "elapsed_sec": elapsed,
            "tokens_per_sec": total_comp_tokens / elapsed,
            "format_compliance_rate": format_compliance_rate,
            "answer_tag_presence_rate": answer_tag_rate,
            "qualitative_samples": qualitative_results,
        }

    def _atomic_save_payload(self, target_path: Path, payload: dict[str, Any]) -> None:
        """Helper to atomically save payload to disk via open file handle and retry replace."""
        import uuid
        temp_path = target_path.parent / f"{target_path.stem}_{uuid.uuid4().hex}.tmp"
        try:
            with open(temp_path, "wb") as f:
                torch.save(payload, f)
                f.flush()
                os.fsync(f.fileno())

            for attempt in range(5):
                try:
                    os.replace(str(temp_path), str(target_path))
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

    def save_sft_checkpoint(
        self,
        val_loss: float = float("inf"),
        is_best: bool = False,
    ) -> Path:
        """Atomically serialize model weights, optimizer, counters, and SFT metadata."""
        ckpt_name = f"step_{self.global_step}.pt"
        ckpt_path = self.checkpoint_dir / ckpt_name

        payload = {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "tokens_seen": self.tokens_seen,
            "completion_tokens_seen": self.completion_tokens_seen,
            "prompt_tokens_seen": self.prompt_tokens_seen,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "condition": self.condition,
            "base_checkpoint_sha256": self.base_checkpoint_hash,
            "tokenizer_sha256": compute_file_sha256(self.tokenizer.model_path),
            "model_config": self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else asdict(self.model.config),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            **_get_rng_states(),
        }

        # 1. Save step checkpoint
        self._atomic_save_payload(ckpt_path, payload)

        # 2. Update latest.pt
        latest_path = self.checkpoint_dir / "latest.pt"
        self._atomic_save_payload(latest_path, payload)

        # 3. Update best.pt if improved
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            self._atomic_save_payload(best_path, payload)

        return ckpt_path

    def load_sft_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        """Restore model, optimizer, scheduler, scaler, counters, and RNG states."""
        path = Path(checkpoint_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        self.global_step = int(ckpt.get("global_step", 0))
        self.current_epoch = int(ckpt.get("epoch", 0))
        self.tokens_seen = int(ckpt.get("tokens_seen", 0))
        self.completion_tokens_seen = int(ckpt.get("completion_tokens_seen", 0))
        self.prompt_tokens_seen = int(ckpt.get("prompt_tokens_seen", 0))
        self.best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

        _restore_rng_states(ckpt)
        self.log_callback(f"Successfully resumed SFT from step {self.global_step} ({path.name})")
        return ckpt
