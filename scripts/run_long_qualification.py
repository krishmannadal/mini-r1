"""
Mini-R1 -- Long-Run Qualification (1,000 Steps)
================================================
Compares the two best learning rate candidates (3e-4 vs 6e-4) across 1,000 steps
under strictly identical seeds, model initializations, and dataset splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import MiniR1Transformer, MiniR1Config
from src.optimizer import create_adamw_optimizer
from src.scheduler import CosineWarmupScheduler
from src.tokenizer import Tokenizer
from src.datasets import ReasoningDatasetManager
from src.torch_dataset import ReasoningTorchDataset
from src.dataloader import create_dataloader
from src.trainer import Trainer


def run_long_candidate(
    candidate_name: str,
    peak_lr: float,
    min_lr: float,
    max_steps: int = 1000,
    warmup_steps: int = 50,
    val_interval: int = 100,
    checkpoint_interval: int = 250,
) -> dict:
    """Execute training for one long qualification candidate and return summary metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_path = PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.model"
    tokenizer = Tokenizer(tok_path)

    # 1. Dataset setup (strictly seeded)
    dataset_mgr = ReasoningDatasetManager(seed=42, val_ratio=0.1)
    train_records = dataset_mgr.get_combined_train(limit_per_dataset=3000, shuffle=True)
    val_records = []
    for d in ("gsm8k", "svamp", "aqua"):
        val_records.extend(dataset_mgr.get_formatted_split(d, split="validation", limit=100))

    model_config = MiniR1Config.from_yaml(PROJECT_ROOT / "configs" / "model.yaml")
    train_dataset = ReasoningTorchDataset(train_records, tokenizer=tokenizer, max_seq_len=model_config.max_seq_len)
    val_dataset = ReasoningTorchDataset(val_records, tokenizer=tokenizer, max_seq_len=model_config.max_seq_len)

    micro_batch = 8
    grad_accum = 4
    effective_batch = micro_batch * grad_accum

    train_loader = create_dataloader(train_dataset, batch_size=micro_batch, shuffle=True, pad_id=tokenizer.pad_id)
    val_loader = create_dataloader(val_dataset, batch_size=micro_batch, shuffle=False, pad_id=tokenizer.pad_id)

    # Distinct isolated directories
    ckpt_dir = PROJECT_ROOT / "artifacts" / f"checkpoints_long_qual_{candidate_name}"
    log_dir = PROJECT_ROOT / "logs" / f"long_qual_{candidate_name}"
    run_id = f"long_qual_{candidate_name}"

    print("\n" + "=" * 80)
    print(f"   STARTING 1000-STEP QUALIFICATION: {candidate_name.upper()} (Peak LR: {peak_lr:.1e})")
    print("=" * 80)
    print(f"  Run ID              : {run_id}")
    print(f"  Model Parameters    : 42.1M")
    print(f"  Effective Batch Size: {effective_batch} (Micro: {micro_batch}, Accum: {grad_accum})")
    print(f"  Precision Mode      : FP16 AMP")
    print(f"  Max Steps           : {max_steps}")
    print(f"  Warmup Steps        : {warmup_steps}")
    print(f"  Validation Interval : Every {val_interval} steps")
    print(f"  Checkpoint Interval : Every {checkpoint_interval} steps")
    print(f"  Checkpoint Dir      : {ckpt_dir}")
    print(f"  Log Dir             : {log_dir}\n")

    # Strictly reset RNG for identical model initialization
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = MiniR1Transformer(model_config).to(device)
    optimizer = create_adamw_optimizer(model, lr=peak_lr, weight_decay=0.01)
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        base_lr=peak_lr,
        min_lr=min_lr,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=model_config,
        scheduler=scheduler,
        device=device,
        amp_dtype="float16" if device == "cuda" else "float32",
        gradient_accumulation_steps=grad_accum,
        max_grad_norm=1.0,
        checkpoint_dir=ckpt_dir,
        log_dir=log_dir,
        run_id=run_id,
        checkpoint_interval_steps=checkpoint_interval,
        val_interval_steps=val_interval,
        log_interval_steps=10,
    )

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        max_steps=max_steps,
        epochs=30,
    )

    # Parse structured logs for analysis
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

    train_losses = [r["train_loss"] for r in step_records if "train_loss" in r]
    grad_norms = [r["grad_norm"] for r in step_records if "grad_norm" in r]
    throughputs = [r["tokens_per_sec_step"] for r in step_records if "tokens_per_sec_step" in r]
    scales = [r["scaler_scale"] for r in step_records if "scaler_scale" in r]
    peak_vrams = [r["gpu_peak_allocated_mb"] for r in step_records if "gpu_peak_allocated_mb" in r]

    val_steps_dict = {r["global_step"]: r["val_loss"] for r in val_records}
    val_ppl_dict = {r["global_step"]: r.get("val_perplexity", 0.0) for r in val_records}

    summary = {
        "candidate": candidate_name,
        "peak_lr": peak_lr,
        "min_lr": min_lr,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_records[-1]["val_loss"] if val_records else None,
        "min_val_loss": min(r["val_loss"] for r in val_records) if val_records else None,
        "best_perplexity": min(r.get("val_perplexity", 9999.0) for r in val_records) if val_records else None,
        "val_loss_at_steps": val_steps_dict,
        "val_perplexity_at_steps": val_ppl_dict,
        "avg_tokens_per_sec": sum(throughputs) / max(1, len(throughputs)),
        "max_peak_vram_mb": max(peak_vrams) if peak_vrams else 0.0,
        "min_scaler_scale": min(scales) if scales else 1.0,
        "skipped_optimizer_steps": trainer.skipped_optimizer_steps,
        "max_grad_norm": max(grad_norms) if grad_norms else 0.0,
        "log_file": str(trainer.log_file),
    }

    return summary


def run_long_qualification(steps: int = 1000) -> dict[str, dict]:
    candidates = [
        ("3e-4", 3e-4, 3e-5),
        ("6e-4", 6e-4, 6e-5),
    ]

    all_summaries = {}
    for name, peak_lr, min_lr in candidates:
        summary = run_long_candidate(
            candidate_name=name,
            peak_lr=peak_lr,
            min_lr=min_lr,
            max_steps=steps,
            warmup_steps=50,
            val_interval=100,
            checkpoint_interval=250,
        )
        all_summaries[name] = summary

    # Save summary JSON
    summary_path = PROJECT_ROOT / "logs" / "long_qualification_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    return all_summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mini-R1 Long-Run Qualification.")
    parser.add_argument("--steps", type=int, default=1000, help="Total optimizer steps per candidate.")
    args = parser.parse_args()

    run_long_qualification(steps=args.steps)
