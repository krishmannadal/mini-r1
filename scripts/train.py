"""
Mini-R1 -- Scaled SLM Training CLI
==================================
Runs safe, mixed-precision (FP16/BF16/FP32), resuable training for the 42.1M parameter
Mini-R1 Transformer on NVIDIA RTX 4050 (6 GB VRAM).

Usage:
    python scripts/train.py --smoke-test
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --resume checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import MiniR1Transformer, MiniR1Config
from src.optimizer import create_adamw_optimizer
from src.scheduler import CosineWarmupScheduler
from src.checkpoint import load_checkpoint
from src.tokenizer import Tokenizer
from src.datasets import ReasoningDatasetManager
from src.torch_dataset import ReasoningTorchDataset
from src.dataloader import create_dataloader
from src.trainer import Trainer


def load_training_config(config_path: str = "configs/train.yaml") -> dict:
    """Load training configuration YAML."""
    path = PROJECT_ROOT / config_path
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("training", data)


def run_smoke_test() -> None:
    """
    Execute smoke test mode:
    1. Trains for 4 optimizer steps with gradient accumulation & mixed precision.
    2. Saves checkpoints/latest.pt.
    3. Resumes from checkpoints/latest.pt without re-initializing weights.
    4. Trains 4 additional steps (step 5 to 8).
    5. Confirms resume consistency.
    """
    print("\n" + "=" * 80)
    print("                 Mini-R1 SLM Training System -- SMOKE TEST")
    print("=" * 80)
    print("  Objective: Verify AMP, Gradient Accumulation, Atomic Checkpointing,")
    print("             and Non-Destructive Resumption on 42.1M SLM.")
    print("=" * 80 + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_path = PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.model"
    if not tok_path.exists():
        print(f"  [ERROR] Tokenizer missing at {tok_path}.")
        sys.exit(1)
    tokenizer = Tokenizer(tok_path)

    # 1. Dataset setup
    dataset_mgr = ReasoningDatasetManager(seed=42, val_ratio=0.1)
    train_records = dataset_mgr.get_combined_train(limit_per_dataset=50, shuffle=True)
    val_records = []
    for d in ("gsm8k", "svamp", "aqua"):
        val_records.extend(dataset_mgr.get_formatted_split(d, split="validation", limit=10))

    model_config = MiniR1Config.from_yaml(PROJECT_ROOT / "configs" / "model.yaml")
    train_dataset = ReasoningTorchDataset(train_records, tokenizer=tokenizer, max_seq_len=model_config.max_seq_len)
    val_dataset = ReasoningTorchDataset(val_records, tokenizer=tokenizer, max_seq_len=model_config.max_seq_len)

    train_loader = create_dataloader(train_dataset, batch_size=4, shuffle=True, pad_id=tokenizer.pad_id)
    val_loader = create_dataloader(val_dataset, batch_size=4, shuffle=False, pad_id=tokenizer.pad_id)

    # 2. Phase 1: Train 4 steps
    smoke_dir = PROJECT_ROOT / "artifacts" / "smoke_checkpoints"
    smoke_logs = PROJECT_ROOT / "artifacts" / "smoke_logs"

    print(">>> PHASE 1: Training Initial 4 Steps...")
    model1 = MiniR1Transformer(model_config).to(device)
    optimizer1 = create_adamw_optimizer(model1, lr=3e-4, weight_decay=0.01)
    scheduler1 = CosineWarmupScheduler(optimizer1, warmup_steps=2, max_steps=10, min_lr=3e-5)

    trainer1 = Trainer(
        model=model1,
        optimizer=optimizer1,
        config=model_config,
        scheduler=scheduler1,
        device=device,
        amp_dtype="float16" if device == "cuda" else "float32",
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        checkpoint_dir=smoke_dir,
        log_dir=smoke_logs,
        run_id="smoke_phase1",
        checkpoint_interval_steps=2,
        val_interval_steps=4,
        log_interval_steps=1,
    )

    history1 = trainer1.train(train_loader=train_loader, val_loader=val_loader, max_steps=4, epochs=5)

    assert trainer1.global_step == 4, f"Expected step 4, got {trainer1.global_step}"
    latest_ckpt = smoke_dir / "latest.pt"
    assert latest_ckpt.exists(), "latest.pt was not saved!"
    print(f"\n[OK] Phase 1 completed. Saved checkpoint at: {latest_ckpt}")

    # 3. Phase 2: Resume and train 4 more steps (to step 8)
    print("\n>>> PHASE 2: Resuming from checkpoint (Steps 5 to 8)...")
    model2 = MiniR1Transformer(model_config).to(device)
    optimizer2 = create_adamw_optimizer(model2, lr=3e-4, weight_decay=0.01)
    scheduler2 = CosineWarmupScheduler(optimizer2, warmup_steps=2, max_steps=10, min_lr=3e-5)

    trainer2 = Trainer(
        model=model2,
        optimizer=optimizer2,
        config=model_config,
        scheduler=scheduler2,
        device=device,
        amp_dtype="float16" if device == "cuda" else "float32",
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        checkpoint_dir=smoke_dir,
        log_dir=smoke_logs,
        run_id="smoke_phase2",
        checkpoint_interval_steps=2,
        val_interval_steps=4,
        log_interval_steps=1,
    )

    history2 = trainer2.train(
        train_loader=train_loader,
        val_loader=val_loader,
        max_steps=8,
        epochs=5,
        resume_from=latest_ckpt,
    )

    assert trainer2.global_step == 8, f"Expected step 8 after resume, got {trainer2.global_step}"

    # Verify checkpoint assertions
    assert not (smoke_dir / "best_model.pt").exists(), "best_model.pt legacy duplicate should not exist!"
    assert (smoke_dir / "best.pt").exists(), "best.pt canonical checkpoint must exist!"
    assert (smoke_dir / "latest.pt").exists(), "latest.pt checkpoint must exist!"
    assert (smoke_dir / "step_8.pt").exists(), "step_8.pt checkpoint must exist!"
    assert len(list(smoke_dir.glob("*.tmp"))) == 0, "No temporary files should exist in checkpoint dir!"

    print("\n" + "=" * 80)
    print("                     SMOKE TEST VERIFICATION RESULTS")
    print("=" * 80)
    print(f"  Phase 1 Start Loss : {history1[0]['train_loss']:.4f}")
    print(f"  Phase 1 End Loss   : {history1[-1]['train_loss']:.4f} (Step 4)")
    print(f"  Phase 2 Resume Step: 4 -> Step {trainer2.global_step}")
    print(f"  Phase 2 End Loss   : {history2[-1]['train_loss']:.4f} (Step 8)")
    print(f"  Checkpoints Created: {[f.name for f in smoke_dir.glob('*.pt')]}")
    print(f"  Phase 1 Log File   : {trainer1.log_file}")
    print(f"  Phase 2 Log File   : {trainer2.log_file}")
    print("  [SUCCESS] All Smoke Test assertions passed seamlessly!")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Mini-R1 Scaled Reasoning Model.")
    parser.add_argument("--smoke-test", action="store_true", help="Run self-contained smoke test verifying training & resume.")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to training config YAML.")
    parser.add_argument("--model-config", type=str, default="configs/model.yaml", help="Path to model config YAML.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt file to resume from.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override total optimization steps.")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    # 1. Load Configurations
    train_cfg = load_training_config(args.config)
    model_cfg = MiniR1Config.from_yaml(PROJECT_ROOT / args.model_config)

    # Merge CLI overrides
    device = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    amp_dtype = train_cfg.get("amp_dtype", "float16")
    micro_batch = train_cfg.get("micro_batch_size", 8)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 4)
    lr = float(train_cfg.get("lr", 3e-4))
    min_lr = float(train_cfg.get("min_lr", 3e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    warmup_steps = int(train_cfg.get("warmup_steps", 20))
    max_steps = args.max_steps or int(train_cfg.get("max_steps", 100))
    epochs = int(train_cfg.get("epochs", 2))
    val_interval = int(train_cfg.get("val_interval_steps", 20))
    ckpt_interval = int(train_cfg.get("checkpoint_interval_steps", 20))
    retained_ckpts = int(train_cfg.get("retained_checkpoints", 3))
    ckpt_dir = train_cfg.get("checkpoint_dir", "checkpoints")
    log_dir = train_cfg.get("log_dir", "logs")

    # 2. Tokenizer
    tok_path = PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.model"
    if not tok_path.exists():
        print(f"[ERROR] Tokenizer missing at {tok_path}.")
        sys.exit(1)
    tokenizer = Tokenizer(tok_path)

    # 3. Datasets
    print("Loading reasoning datasets (GSM8K, SVAMP, AQuA)...")
    dataset_mgr = ReasoningDatasetManager(seed=42, val_ratio=0.1)
    train_records = dataset_mgr.get_combined_train(
        limit_per_dataset=train_cfg.get("max_train_samples", 3000) // 3,
        shuffle=True,
    )
    val_records = []
    for d in ("gsm8k", "svamp", "aqua"):
        val_records.extend(dataset_mgr.get_formatted_split(d, split="validation", limit=train_cfg.get("max_val_samples", 300) // 3))

    train_dataset = ReasoningTorchDataset(train_records, tokenizer=tokenizer, max_seq_len=model_cfg.max_seq_len)
    val_dataset = ReasoningTorchDataset(val_records, tokenizer=tokenizer, max_seq_len=model_cfg.max_seq_len)

    train_loader = create_dataloader(train_dataset, batch_size=micro_batch, shuffle=True, pad_id=tokenizer.pad_id)
    val_loader = create_dataloader(val_dataset, batch_size=micro_batch, shuffle=False, pad_id=tokenizer.pad_id)

    # 4. Model, Optimizer, Scheduler
    model = MiniR1Transformer(model_cfg).to(device)
    optimizer = create_adamw_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_lr=min_lr,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=model_cfg,
        scheduler=scheduler,
        device=device,
        amp_dtype=amp_dtype,
        gradient_accumulation_steps=grad_accum,
        max_grad_norm=max_grad_norm,
        checkpoint_dir=ckpt_dir,
        log_dir=log_dir,
        retained_checkpoints=retained_ckpts,
        val_interval_steps=val_interval,
        checkpoint_interval_steps=ckpt_interval,
        log_interval_steps=5,
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        max_steps=max_steps,
        epochs=epochs,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
