"""
Mini-R1 -- Controlled Direct-Answer vs CoT SFT Training Runner (Task 15C)
========================================================================
Executes full supervised fine-tuning training runs for:
1. Direct-Answer SFT (Control Condition)
2. Chain-of-Thought SFT (Treatment Condition)

Under strictly identical hyperparameters:
- Base Checkpoint: best.pt (step 18,000, val loss 2.8726)
- Optimizer: AdamW (lr=1e-4, min_lr=1e-5, warmup=50 steps, cosine decay)
- Effective Batch: 32 (micro=8, accum=4)
- Optimizer Steps: 650 steps (~3.0 epochs over 6,973 GSM8K problems)
- Dev Evaluation: Every 50 steps on GSM8K Dev (500 problems)
- Artifact Export: artifacts/sft/direct/ and artifacts/sft/cot/
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any
import yaml
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set UTF-8 encoding for stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.model import MiniR1Transformer, MiniR1Config
from src.sft_dataset import SFTDataset, collate_sft_batch
from src.sft_trainer import SFTTrainer, compute_file_sha256
from src.tokenizer import Tokenizer


TOKENIZER_PATH = PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.model"
BASE_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "mini_r1_500m_pretrain" / "best.pt"
BASE_CHECKPOINT_SHA256 = "f058a558edc2020b1a4a7a7b54312b97fb0b0a20e16c42da92018686e66462a4"


def run_single_sft_branch(
    condition: str,
    config_path: Path,
    tokenizer: Tokenizer,
    device: str = "cuda",
) -> dict[str, Any]:
    """Train a single SFT branch (direct or cot) for exactly max_steps optimizer steps."""
    print(f"\n{'='*70}")
    print(f" LAUNCHING SFT EXPERIMENT: [{condition.upper()}]")
    print(f"{'='*70}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Base checkpoint integrity check
    pre_hash = compute_file_sha256(BASE_CHECKPOINT_PATH)
    assert pre_hash == BASE_CHECKPOINT_SHA256, f"Base checkpoint integrity error: {pre_hash} != {BASE_CHECKPOINT_SHA256}"

    # 2. Load base model
    ckpt = torch.load(BASE_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model_cfg = MiniR1Config.from_dict(ckpt.get("config", {}))
    model = MiniR1Transformer(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])

    # 3. Create datasets & dataloaders
    train_ds = SFTDataset(
        jsonl_path=PROJECT_ROOT / cfg["data"]["train_path"],
        tokenizer=tokenizer,
        condition=condition,
        max_seq_len=cfg["data"]["max_seq_len"],
        add_eos=cfg["data"]["add_eos"],
    )
    dev_ds = SFTDataset(
        jsonl_path=PROJECT_ROOT / cfg["data"]["dev_path"],
        tokenizer=tokenizer,
        condition=condition,
        max_seq_len=cfg["data"]["max_seq_len"],
        add_eos=cfg["data"]["add_eos"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["micro_batch_size"],
        shuffle=True,
        collate_fn=collate_sft_batch,
        generator=torch.Generator().manual_seed(cfg["experiment"]["seed"]),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg["training"]["micro_batch_size"],
        shuffle=False,
        collate_fn=collate_sft_batch,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=cfg,
        device=device,
        base_checkpoint_path=BASE_CHECKPOINT_PATH,
    )

    max_steps = trainer.max_steps
    val_interval = trainer.val_interval_steps
    ckpt_interval = trainer.checkpoint_interval_steps
    log_interval = trainer.log_interval_steps

    print(f"Target Steps: {max_steps} | Dev Validation Every: {val_interval} steps | Checkpoint Every: {ckpt_interval} steps")
    print(f"Train Dataset: {len(train_ds):,} examples | Dev Dataset: {len(dev_ds):,} examples")

    training_logs: list[dict[str, Any]] = []
    dev_eval_logs: list[dict[str, Any]] = []
    skipped_steps_count = 0

    # 4. Step 0 Dev Baseline Evaluation
    print(f"\n[Step 0 Baseline Dev Evaluation - {condition.upper()}]...")
    dev_0 = trainer.evaluate_dev(dev_loader, num_qualitative_samples=5)
    dev_0["step"] = 0
    dev_0["timestamp"] = datetime.now(timezone.utc).isoformat()
    dev_eval_logs.append(dev_0)
    trainer.best_val_loss = dev_0["eval_loss"]

    # Initial checkpoint at step 0
    trainer.save_sft_checkpoint(val_loss=dev_0["eval_loss"], is_best=True)
    print(f"Step 000 | Dev Loss: {dev_0['eval_loss']:.4f} | Dev PPL: {dev_0['perplexity']:.2f} | Format: {dev_0['format_compliance_rate']*100:.1f}%")

    # 5. Training Loop
    data_iter = iter(train_loader)
    epoch = 0
    start_train_time = time.perf_counter()

    while trainer.global_step < max_steps:
        opt_step = trainer.global_step + 1
        step_loss_sum = 0.0
        step_comp_tokens = 0
        step_total_tokens = 0
        step_prompt_tokens = 0
        t0 = time.perf_counter()

        for micro in range(trainer.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                trainer.current_epoch = epoch
                data_iter = iter(train_loader)
                batch = next(data_iter)

            is_accum = (micro < trainer.grad_accum_steps - 1)
            loss_val, comp_toks, total_toks = trainer.train_step(batch, is_accumulating=is_accum)
            step_loss_sum += loss_val
            step_comp_tokens += comp_toks
            step_total_tokens += total_toks
            step_prompt_tokens += int(batch["prompt_tokens_count"].sum().item())

        grad_norm, success = trainer.optimizer_step()
        elapsed_step = max(1e-5, time.perf_counter() - t0)
        avg_step_loss = step_loss_sum / trainer.grad_accum_steps
        current_lr = trainer.optimizer.param_groups[0]["lr"]

        if not success:
            skipped_steps_count += 1
            print(f"[WARN] Step {opt_step} skipped due to non-finite gradient norm ({grad_norm})")

        scaler_scale = trainer.scaler.get_scale() if trainer.scaler.is_enabled() else 1.0
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

        step_record = {
            "step": trainer.global_step,
            "epoch": epoch,
            "loss": avg_step_loss,
            "grad_norm": grad_norm,
            "learning_rate": current_lr,
            "scaler_scale": scaler_scale,
            "step_comp_tokens": step_comp_tokens,
            "step_total_tokens": step_total_tokens,
            "cumulative_tokens": trainer.tokens_seen,
            "cumulative_comp_tokens": trainer.completion_tokens_seen,
            "cumulative_prompt_tokens": trainer.prompt_tokens_seen,
            "tokens_per_sec": step_total_tokens / elapsed_step,
            "elapsed_sec": elapsed_step,
            "peak_vram_mb": peak_vram_mb,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        training_logs.append(step_record)

        # Write to JSONL stream
        with open(trainer.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(step_record) + "\n")

        if trainer.global_step % log_interval == 0 or trainer.global_step == max_steps:
            print(
                f"Step {trainer.global_step:03d}/{max_steps} (Ep {epoch}) | "
                f"Loss: {avg_step_loss:.4f} | Grad: {grad_norm:.3f} | LR: {current_lr:.2e} | "
                f"CompTok: {step_comp_tokens} | Tok/s: {step_record['tokens_per_sec']:.0f} | VRAM: {peak_vram_mb:.0f}MB"
            )

        # Periodic Dev Evaluation
        if trainer.global_step % val_interval == 0 or trainer.global_step == max_steps:
            dev_res = trainer.evaluate_dev(dev_loader, num_qualitative_samples=5)
            dev_res["step"] = trainer.global_step
            dev_res["epoch"] = epoch
            dev_res["timestamp"] = datetime.now(timezone.utc).isoformat()
            dev_eval_logs.append(dev_res)

            is_best = (dev_res["eval_loss"] < trainer.best_val_loss)
            if is_best:
                trainer.best_val_loss = dev_res["eval_loss"]

            saved_ckpt = trainer.save_sft_checkpoint(val_loss=dev_res["eval_loss"], is_best=is_best)
            best_tag = " [NEW BEST]" if is_best else ""
            print(
                f"--> [DEV EVAL Step {trainer.global_step:03d}] Loss: {dev_res['eval_loss']:.4f} | "
                f"PPL: {dev_res['perplexity']:.2f} | Format: {dev_res['format_compliance_rate']*100:.1f}%{best_tag}"
            )

    total_elapsed = max(1e-5, time.perf_counter() - start_train_time)
    print(f"\nTraining for [{condition.upper()}] completed in {total_elapsed/60:.2f} minutes.")
    print(f"Final Global Step: {trainer.global_step} | Best Dev Loss: {trainer.best_val_loss:.4f} | Skipped Steps: {skipped_steps_count}")

    # 6. Export immutable artifacts to artifacts/sft/<condition>/
    artifact_dir = PROJECT_ROOT / "artifacts" / "sft" / condition
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Copy best.pt, latest.pt, logs, and config
    shutil.copy2(trainer.checkpoint_dir / "best.pt", artifact_dir / "best.pt")
    shutil.copy2(trainer.checkpoint_dir / "latest.pt", artifact_dir / "latest.pt")
    shutil.copy2(trainer.log_file, artifact_dir / "sft_train_log.jsonl")
    shutil.copy2(config_path, artifact_dir / "config.yaml")

    with open(artifact_dir / "dev_eval_history.json", "w", encoding="utf-8") as f:
        json.dump(dev_eval_logs, f, indent=2)

    manifest = {
        "condition": condition,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "total_steps": trainer.global_step,
        "total_epochs": epoch,
        "best_dev_loss": trainer.best_val_loss,
        "total_tokens_seen": trainer.tokens_seen,
        "completion_tokens_seen": trainer.completion_tokens_seen,
        "prompt_tokens_seen": trainer.prompt_tokens_seen,
        "skipped_steps": skipped_steps_count,
        "elapsed_sec": total_elapsed,
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "best_checkpoint_sha256": compute_file_sha256(artifact_dir / "best.pt"),
        "latest_checkpoint_sha256": compute_file_sha256(artifact_dir / "latest.pt"),
    }
    with open(artifact_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Exported immutable artifacts to: {artifact_dir}")

    # 7. Post-training base checkpoint immutability check
    post_hash = compute_file_sha256(BASE_CHECKPOINT_PATH)
    assert post_hash == BASE_CHECKPOINT_SHA256, f"BASE CHECKPOINT CORRUPTED! {post_hash}"

    return {
        "condition": condition,
        "manifest": manifest,
        "dev_eval_logs": dev_eval_logs,
        "training_logs": training_logs,
        "artifact_dir": str(artifact_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Mini-R1 SFT Experiment (Task 15C)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--only", type=str, choices=["all", "direct", "cot"], default="all")
    args = parser.parse_args()

    print(f"Mini-R1 Task 15C SFT Training Orchestrator")
    print(f"Device: {args.device} | Base Checkpoint SHA256: {BASE_CHECKPOINT_SHA256[:12]}...")

    tokenizer = Tokenizer(TOKENIZER_PATH)
    results = {}

    if args.only in ("all", "direct"):
        direct_cfg = PROJECT_ROOT / "configs" / "sft_direct.yaml"
        results["direct"] = run_single_sft_branch("direct", direct_cfg, tokenizer, device=args.device)

    if args.only in ("all", "cot"):
        cot_cfg = PROJECT_ROOT / "configs" / "sft_cot.yaml"
        results["cot"] = run_single_sft_branch("cot", cot_cfg, tokenizer, device=args.device)

    print("\n" + "="*70)
    print(" ALL SFT TRAINING BRANCHES FINISHED SUCCESSFULLY")
    print("="*70)


if __name__ == "__main__":
    main()
