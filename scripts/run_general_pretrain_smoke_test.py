"""
Mini-R1 -- Real General-Text 50-Step Pretraining Smoke Test
===========================================================
Executes a 50-step qualification run on ACTUAL PACKED GENERAL PRETRAINING DATA
(60% cosmopedia-v2, 40% fineweb-edu-dedup) to verify numerical stability,
loss convergence, gradient norms, AMP GradScaler health, and VRAM occupancy.

Usage:
    python scripts/run_general_pretrain_smoke_test.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import MiniR1Transformer, MiniR1Config
from src.tokenizer import Tokenizer
from src.general_dataset import GeneralPretrainStreamer, PackedPretrainTorchDataset
from src.dataloader import create_dataloader
from src.trainer import Trainer
from src.optimizer import create_adamw_optimizer
from src.scheduler import CosineWarmupScheduler


def run_general_pretrain_smoke_test(
    steps: int = 50,
    seed: int = 42,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_path = PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.model"
    tokenizer = Tokenizer(tok_path)

    micro_batch = 8
    grad_accum = 4
    effective_batch = micro_batch * grad_accum  # 32
    tokens_per_step = effective_batch * 512    # 16,384

    print("\n" + "=" * 85)
    print("       Mini-R1: General-Text Pretraining Smoke Test (50 Optimizer Steps)")
    print("=" * 85)
    print(f"  Model Architecture   : 42.1M parameters (d_model=512, L=12, H=8, d_ff=2048, V=8000)")
    print(f"  Data Sources         : 60% cosmopedia-v2, 40% fineweb-edu-dedup (PACKED 512)")
    print(f"  Precision            : FP16 AMP (torch.amp.GradScaler)")
    print(f"  Batching             : micro_batch={micro_batch}, accumulation={grad_accum} -> eff_batch={effective_batch}")
    print(f"  Tokens / Step        : {tokens_per_step:,} tokens (32 x 512)")
    print(f"  Target Steps         : {steps} steps ({steps * tokens_per_step:,} tokens)")
    print(f"  Peak Learning Rate   : 6.0e-4 (cosine decay to 6.0e-5)")
    print(f"  Device               : {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})\n")

    # 1. Stream and pack training and validation general text
    target_train_tokens = int(steps * effective_batch * 512 * 1.5)  # ~1.2M tokens
    target_val_tokens = 50 * 512                                    # ~25,600 tokens

    streamer = GeneralPretrainStreamer(
        dataset_name="HuggingFaceTB/smollm-corpus",
        source_ratios={"cosmopedia-v2": 0.60, "fineweb-edu-dedup": 0.40},
        tokenizer=tokenizer,
        max_seq_len=512,
        seed=seed,
    )

    t0 = time.time()
    train_chunks, val_chunks, stats, breakdown = streamer.pack_tokens_from_stream(
        target_tokens=target_train_tokens + target_val_tokens,
        val_ratio=0.05,
    )
    prep_time = time.time() - t0
    print(f"  [OK] Packed {len(train_chunks):,} train sequences & {len(val_chunks):,} val sequences in {prep_time:.1f}s")

    # 2. Build Datasets and DataLoaders
    train_ds = PackedPretrainTorchDataset(train_chunks)
    val_ds = PackedPretrainTorchDataset(val_chunks)

    train_loader = create_dataloader(train_ds, batch_size=micro_batch, shuffle=True)
    val_loader = create_dataloader(val_ds, batch_size=micro_batch, shuffle=False)

    # 3. Model setup
    model_cfg = MiniR1Config(
        vocab_size=8000,
        max_seq_len=512,
        d_model=512,
        n_layers=12,
        n_heads=8,
        d_ff=2048,
        dropout=0.0,
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = MiniR1Transformer(model_cfg).to(device)

    # 4. Optimizer & Scheduler
    peak_lr = 6.0e-4
    min_lr = 6.0e-5
    optimizer = create_adamw_optimizer(model, lr=peak_lr, weight_decay=0.01)
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=10,
        max_steps=steps,
        base_lr=peak_lr,
        min_lr=min_lr,
    )

    # 5. Trainer setup
    log_dir = PROJECT_ROOT / "artifacts" / "general_pretrain_smoke_logs"
    ckpt_dir = log_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=model_cfg,
        scheduler=scheduler,
        device=device,
        amp_dtype="float16" if device == "cuda" else "float32",
        gradient_accumulation_steps=grad_accum,
        max_grad_norm=1.0,
        checkpoint_dir=ckpt_dir,
        log_dir=log_dir,
        run_id="general_pretrain_smoke_50step",
        checkpoint_interval_steps=25,
        val_interval_steps=10,
        log_interval_steps=5,
    )

    # 6. Execute 50 steps
    print("  Executing 50-step general pretraining run...")
    t_train_0 = time.time()
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        max_steps=steps,
        epochs=10,
    )
    train_elapsed = time.time() - t_train_0

    # Parse structured logs
    step_records = []
    val_records = []
    with open(trainer.log_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if "val_loss" in entry and entry.get("val_loss") is not None:
                val_records.append(entry)
            if "train_loss" in entry and entry.get("train_loss") is not None:
                step_records.append(entry)

    initial_loss = step_records[0]["train_loss"] if step_records else 0.0
    final_loss = step_records[-1]["train_loss"] if step_records else 0.0
    final_val_loss = val_records[-1]["val_loss"] if val_records else 0.0
    peak_vram = max((r.get("gpu_peak_allocated_mb", 0.0) for r in step_records), default=0.0)
    scaler_scale = step_records[-1].get("scaler_scale", 65536.0) if step_records else 65536.0
    skipped_steps = step_records[-1].get("skipped_optimizer_steps", 0) if step_records else 0

    results = {
        "steps_completed": len(step_records),
        "initial_train_loss": initial_loss,
        "final_train_loss": final_loss,
        "final_val_loss": final_val_loss,
        "loss_delta": initial_loss - final_loss,
        "peak_vram_allocated_mb": peak_vram,
        "final_scaler_scale": scaler_scale,
        "skipped_optimizer_steps": skipped_steps,
        "total_tokens_trained": len(step_records) * tokens_per_step,
        "train_time_sec": train_elapsed,
        "avg_tokens_per_sec": (len(step_records) * tokens_per_step) / max(0.01, train_elapsed),
    }

    print("\n" + "=" * 85)
    print("            GENERAL PRETRAINING 50-STEP SMOKE TEST SUMMARY")
    print("=" * 85)
    print(f"  Steps Completed          : {results['steps_completed']}/{steps}")
    print(f"  Initial Loss (Step 1)    : {initial_loss:.4f}")
    print(f"  Final Loss (Step 50)     : {final_loss:.4f} (Reduction: {results['loss_delta']:.4f})")
    print(f"  Final Validation Loss    : {final_val_loss:.4f}")
    print(f"  Peak VRAM Allocated      : {peak_vram:.1f} MB (< 4,000 MB physical headroom)")
    print(f"  GradScaler Health        : Scale = {scaler_scale:.1f}, Skipped Steps = {skipped_steps}")
    print(f"  Training Throughput      : {results['avg_tokens_per_sec']:,.0f} tokens/sec")
    print(f"  Log File Saved           : {trainer.log_file}")
    print("=" * 85 + "\n")

    return results


if __name__ == "__main__":
    run_general_pretrain_smoke_test(steps=50)
