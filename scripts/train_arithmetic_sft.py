"""
train_arithmetic_sft.py - Unified Masked SFT Trainer for Task 18B.

Supports:
1. Compact Arithmetic SFT (--condition compact)
2. Verbose Arithmetic SFT (--condition verbose)

Uses:
- Base checkpoint: checkpoints/mini_r1_500m_pretrain/best.pt (18,000 steps)
- Completion-only loss masking (label = -100 for prompt tokens)
- FP16 AMP with GradScaler
- AdamW (LR=1e-4, weight_decay=0.01, max_grad_norm=1.0, cosine schedule)
- Periodic validation on arithmetic dev split
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import MiniR1Config, MiniR1Transformer
from src.tokenizer import Tokenizer


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class MaskedArithmeticSFTDataset(Dataset):
    """Dataset for causal LM SFT with prompt token masking."""

    def __init__(self, jsonl_path: str, tokenizer: Tokenizer, max_seq_len: int = 256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: list[dict[str, Any]] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.examples[idx]
        prompt = rec["prompt"]
        completion = rec["completion"]

        prompt_toks = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        completion_toks = self.tokenizer.encode(completion, add_bos=False, add_eos=True)

        full_seq = prompt_toks + completion_toks
        if len(full_seq) > self.max_seq_len:
            full_seq = full_seq[: self.max_seq_len]

        input_ids = full_seq[:-1]
        target_ids = full_seq[1:]

        # Mask prompt tokens in target: prompt_toks length minus 1
        num_prompt_targets = max(0, len(prompt_toks) - 1)
        labels = list(target_ids)
        for i in range(min(num_prompt_targets, len(labels))):
            labels[i] = -100

        # Active completion tokens count
        active_completion_toks = sum(1 for x in labels if x != -100)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_tokens": len(prompt_toks),
            "completion_tokens": len(completion_toks),
            "active_completion_tokens": active_completion_toks,
        }


def collate_sft_batch(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor | int]:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids_list = []
    labels_list = []
    tot_prompt_toks = 0
    tot_comp_toks = 0
    tot_active_comp_toks = 0

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len
        input_ids_list.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
        labels_list.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))

        tot_prompt_toks += item["prompt_tokens"]
        tot_comp_toks += item["completion_tokens"]
        tot_active_comp_toks += item["active_completion_tokens"]

    return {
        "input_ids": torch.stack(input_ids_list, dim=0),
        "labels": torch.stack(labels_list, dim=0),
        "total_prompt_tokens": tot_prompt_toks,
        "total_completion_tokens": tot_comp_toks,
        "total_active_completion_tokens": tot_active_comp_toks,
    }


def evaluate_dev_loss(model: MiniR1Transformer, dev_loader: DataLoader, device: str) -> tuple[float, float]:
    """Evaluate masked SFT loss and perplexity on dev set."""
    model.eval()
    total_loss = 0.0
    total_active_toks = 0

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    with torch.inference_mode():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16):
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            active_toks = (labels != -100).sum().item()
            total_loss += loss.item()
            total_active_toks += active_toks

    avg_loss = total_loss / max(1, total_active_toks)
    ppl = math.exp(min(20.0, avg_loss))
    model.train()
    return avg_loss, ppl


def get_lr(step: int, warmup_steps: int, total_steps: int, peak_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return peak_lr * float(step) / float(max(1, warmup_steps))
    if step > total_steps:
        return min_lr
    decay_ratio = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (peak_lr - min_lr)


def main():
    parser = argparse.ArgumentParser(description="Train Verified Arithmetic SFT Model (Task 18B)")
    parser.add_argument("--condition", type=str, required=True, choices=["compact", "verbose"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--peak_lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=75)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(f" MINI-R1 TASK 18B: SFT TRAINING — CONDITION: {args.condition.upper()}")
    print(f" Device: {device} | Seed: {args.seed} | Micro Batch: {args.micro_batch_size} | Grad Accum: {args.grad_accum_steps}")
    print(f" Effective Batch Size: {args.micro_batch_size * args.grad_accum_steps}")
    print("=" * 70)

    # Paths
    base_ckpt_path = "checkpoints/mini_r1_500m_pretrain/best.pt"
    tok_path = "artifacts/tokenizer/tokenizer.model"
    manifest_path = "data/arithmetic_sft/manifest.json"

    train_jsonl = f"data/arithmetic_sft/{args.condition}_train.jsonl"
    dev_jsonl = f"data/arithmetic_sft/{args.condition}_dev.jsonl"
    out_dir = f"artifacts/sft/arithmetic_{args.condition}"
    os.makedirs(out_dir, exist_ok=True)

    base_sha = compute_file_sha256(base_ckpt_path)
    tok_sha = compute_file_sha256(tok_path)
    manifest_sha = compute_file_sha256(manifest_path)

    print(f"Base Checkpoint SHA256: {base_sha}")
    print(f"Tokenizer SHA256:       {tok_sha}")
    print(f"Manifest SHA256:        {manifest_sha}")

    # Load Tokenizer and Datasets
    tokenizer = Tokenizer(tok_path)
    train_dataset = MaskedArithmeticSFTDataset(train_jsonl, tokenizer, max_seq_len=256)
    dev_dataset = MaskedArithmeticSFTDataset(dev_jsonl, tokenizer, max_seq_len=256)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_sft_batch(b, pad_id=tokenizer.pad_id),
        generator=torch.Generator().manual_seed(args.seed),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.micro_batch_size * 2,
        shuffle=False,
        collate_fn=lambda b: collate_sft_batch(b, pad_id=tokenizer.pad_id),
    )

    # Load Model
    base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = base_ckpt.get("model_config", base_ckpt.get("config", {}))
    model_cfg = MiniR1Config.from_dict(cfg_dict)
    model = MiniR1Transformer(model_cfg).to(device)
    model.load_state_dict(base_ckpt["model_state_dict"])
    print(f"Loaded Base Checkpoint (Step: {base_ckpt.get('step', 18000)}, Params: {model.get_num_params():,})")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=("cuda" in device))
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Steps calculation
    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    total_optimizer_steps = (args.epochs * steps_per_epoch) if not args.smoke_test else 10
    warmup_steps = int(0.05 * total_optimizer_steps) if not args.smoke_test else 2

    print(f"Total Steps: {total_optimizer_steps} | Warmup Steps: {warmup_steps} | Epochs: {args.epochs}")

    # Baseline Dev Loss
    init_dev_loss, init_dev_ppl = evaluate_dev_loss(model, dev_loader, device)
    print(f"Initial Zero-Shot Dev Loss: {init_dev_loss:.4f} | PPL: {init_dev_ppl:.2f}")

    best_dev_loss = float("inf")
    best_dev_ppl = float("inf")
    best_step = 0

    train_logs: list[dict[str, Any]] = []
    global_step = 0
    total_tokens_seen = 0
    completion_tokens_seen = 0
    prompt_tokens_seen = 0
    skipped_steps = 0

    t_start = time.perf_counter()
    epoch = 0

    while global_step < total_optimizer_steps:
        epoch += 1
        model.train()
        accum_loss = 0.0
        micro_step = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16):
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss_scaled = loss / args.grad_accum_steps

            scaler.scale(loss_scaled).backward()
            accum_loss += loss.item()
            micro_step += 1

            total_tokens_seen += input_ids.numel()
            prompt_tokens_seen += batch["total_prompt_tokens"]
            completion_tokens_seen += batch["total_completion_tokens"]

            if micro_step % args.grad_accum_steps == 0:
                global_step += 1
                curr_lr = get_lr(global_step, warmup_steps, total_optimizer_steps, args.peak_lr, args.min_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = curr_lr

                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item()

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                optimizer.zero_grad(set_to_none=True)

                if scale_after < scale_before:
                    skipped_steps += 1

                step_loss = accum_loss / args.grad_accum_steps
                accum_loss = 0.0

                # Logging
                if global_step % 25 == 0 or global_step == 1 or global_step == total_optimizer_steps:
                    elapsed = time.perf_counter() - t_start
                    tok_per_sec = total_tokens_seen / max(1e-5, elapsed)
                    vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
                    print(
                        f"  [Step {global_step:04d}/{total_optimizer_steps:04d}] "
                        f"Loss: {step_loss:.4f} | LR: {curr_lr:.2e} | Grad: {grad_norm:.3f} | "
                        f"Comp Toks: {completion_tokens_seen:,} | Tok/s: {tok_per_sec:.0f} | VRAM: {vram_mb:.0f}MB",
                        flush=True,
                    )

                # Validation
                if global_step % args.eval_interval == 0 or global_step == total_optimizer_steps:
                    dev_loss, dev_ppl = evaluate_dev_loss(model, dev_loader, device)
                    print(f"  >>> [EVAL @ Step {global_step}] Dev Loss: {dev_loss:.4f} | Dev PPL: {dev_ppl:.2f}", flush=True)

                    is_best = dev_loss < best_dev_loss
                    if is_best:
                        best_dev_loss = dev_loss
                        best_dev_ppl = dev_ppl
                        best_step = global_step
                        best_path = os.path.join(out_dir, "best.pt")
                        torch.save({
                            "step": global_step,
                            "model_state_dict": model.state_dict(),
                            "model_config": model_cfg.to_dict(),
                            "dev_loss": dev_loss,
                            "dev_ppl": dev_ppl,
                            "condition": args.condition,
                            "base_checkpoint_sha256": base_sha,
                            "tokenizer_sha256": tok_sha,
                            "manifest_sha256": manifest_sha,
                        }, best_path)
                        print(f"      Saved new best checkpoint to: {best_path}")

                    train_logs.append({
                        "step": global_step,
                        "train_loss": step_loss,
                        "dev_loss": dev_loss,
                        "dev_ppl": dev_ppl,
                        "lr": curr_lr,
                        "grad_norm": grad_norm,
                        "total_tokens": total_tokens_seen,
                        "completion_tokens": completion_tokens_seen,
                        "prompt_tokens": prompt_tokens_seen,
                        "skipped_steps": skipped_steps,
                    })

                if global_step >= total_optimizer_steps:
                    break

    # Save Final Checkpoint
    final_path = os.path.join(out_dir, "final.pt")
    torch.save({
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "model_config": model_cfg.to_dict(),
        "dev_loss": dev_loss if 'dev_loss' in locals() else step_loss,
        "condition": args.condition,
        "base_checkpoint_sha256": base_sha,
        "tokenizer_sha256": tok_sha,
        "manifest_sha256": manifest_sha,
    }, final_path)

    # Save Config & Training Logs
    config_record = {
        "condition": args.condition,
        "seed": args.seed,
        "epochs": args.epochs,
        "optimizer_steps": global_step,
        "peak_lr": args.peak_lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "effective_batch_size": args.micro_batch_size * args.grad_accum_steps,
        "best_step": best_step,
        "best_dev_loss": best_dev_loss,
        "best_dev_ppl": best_dev_ppl,
        "total_tokens_seen": total_tokens_seen,
        "completion_tokens_seen": completion_tokens_seen,
        "prompt_tokens_seen": prompt_tokens_seen,
        "base_checkpoint_sha256": base_sha,
        "tokenizer_sha256": tok_sha,
        "manifest_sha256": manifest_sha,
        "elapsed_seconds": time.perf_counter() - t_start,
    }

    with open(os.path.join(out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)

    with open(os.path.join(out_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(train_logs, f, indent=2)

    print("\n" + "=" * 70)
    print(f" TRAINING COMPLETE FOR {args.condition.upper()}")
    print(f" Best Step: {best_step} | Best Dev Loss: {best_dev_loss:.4f} | Best Dev PPL: {best_dev_ppl:.2f}")
    print(f" Completion Tokens Seen: {completion_tokens_seen:,} | Total Tokens: {total_tokens_seen:,}")
    print(f" Saved to: {out_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
