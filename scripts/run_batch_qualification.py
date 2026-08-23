"""
Mini-R1 -- Real Batch Qualification Experiment (50 Steps)
==========================================================
Validates that the full-scale 42.1M parameter SLM trains safely with
micro_batch=8, accumulation=4 (effective batch=32, 16,384 tokens/step, context=512)
under FP16 AMP on the NVIDIA RTX 4050 (6 GB VRAM).
"""

from __future__ import annotations

import argparse
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


def run_batch_qualification(steps: int = 50) -> None:
    print("\n" + "=" * 80)
    print("        Mini-R1: Real Batch Qualification (42.1M SLM, Effective Batch 32)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("  [WARNING] CUDA device not detected. Running on CPU.")

    tok_path = PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.model"
    if not tok_path.exists():
        print(f"  [ERROR] Tokenizer missing at {tok_path}")
        sys.exit(1)
    tokenizer = Tokenizer(tok_path)

    # 1. Dataset setup
    dataset_mgr = ReasoningDatasetManager(seed=42, val_ratio=0.1)
    train_records = dataset_mgr.get_combined_train(limit_per_dataset=1000, shuffle=True)
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

    # 2. Output directories
    qual_dir = PROJECT_ROOT / "artifacts" / "checkpoints_batch50"
    log_dir = PROJECT_ROOT / "logs" / "qualification_batch50"

    print(f"  Model Parameters    : 42.1M")
    print(f"  Context Length      : {model_config.max_seq_len}")
    print(f"  Micro Batch Size    : {micro_batch}")
    print(f"  Accumulation Steps  : {grad_accum}")
    print(f"  Effective Batch Size: {effective_batch} sequences ({effective_batch * model_config.max_seq_len:,} tokens/step)")
    print(f"  Target Steps        : {steps}")
    print(f"  Validation Interval : 25 steps")
    print(f"  Checkpoints Dir     : {qual_dir}")
    print(f"  Logs Dir            : {log_dir}\n")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = MiniR1Transformer(model_config).to(device)
    optimizer = create_adamw_optimizer(model, lr=3e-4, weight_decay=0.01)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=10, max_steps=steps, min_lr=3e-5)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=model_config,
        scheduler=scheduler,
        device=device,
        amp_dtype="float16" if device == "cuda" else "float32",
        gradient_accumulation_steps=grad_accum,
        max_grad_norm=1.0,
        checkpoint_dir=qual_dir,
        log_dir=log_dir,
        run_id="qual_batch50_real",
        checkpoint_interval_steps=25,
        val_interval_steps=25,
        log_interval_steps=1,
    )

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        max_steps=steps,
        epochs=10,
    )

    print("\n" + "=" * 80)
    print("           REAL BATCH QUALIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"  Final Global Step  : {trainer.global_step}")
    print(f"  Best Val Loss      : {trainer.best_val_loss:.4f}")
    print(f"  Log File           : {trainer.log_file}")
    print(f"  Checkpoints Saved  : {[f.name for f in qual_dir.glob('*.pt')]}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Real Batch Qualification.")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    run_batch_qualification(steps=args.steps)
