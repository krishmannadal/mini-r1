"""
train_subtraction_sft.py - SFT Training Engine for Task 19C.

Trains a clean, single-step subtraction SFT model on [50, 99] operands.
- Base Checkpoint: checkpoints/mini_r1_500m_pretrain/best.pt (18,000 steps)
- Completion-only loss masking
- AdamW (peak LR=1e-4, min LR=1e-5, weight_decay=0.01, max_grad_norm=1.0, cosine schedule)
- Effective batch size: 32 (micro_batch=8, grad_accum=4)
- Epochs: 3 (1,125 optimizer steps)
- Evaluates dev loss every 75 steps and saves best checkpoint
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


class MaskedSubtractionSFTDataset(Dataset):
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

        num_prompt_targets = max(0, len(prompt_toks) - 1)
        labels = list(target_ids)
        for i in range(min(num_prompt_targets, len(labels))):
            labels[i] = -100

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

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len
        input_ids_list.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
        labels_list.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(input_ids_list, dim=0),
        "labels": torch.stack(labels_list, dim=0),
        "total_prompt_tokens": sum(x["prompt_tokens"] for x in batch),
        "total_completion_tokens": sum(x["completion_tokens"] for x in batch),
        "active_completion_tokens": sum(x["active_completion_tokens"] for x in batch),
    }


def get_lr(step: int, warmup_steps: int, total_steps: int, peak_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.inference_mode()
def evaluate_dev_loss(model: MiniR1Transformer, dataloader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16):
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        valid_tokens = (labels != -100).sum().item()
        total_loss += loss.item()
        total_tokens += valid_tokens

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser(description="Task 19C Clean Subtraction SFT Trainer")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--peak_lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_ckpt_path = "checkpoints/mini_r1_500m_pretrain/best.pt"
    tok_path = "artifacts/tokenizer/tokenizer.model"
    train_jsonl = "data/subtraction_sft/train.jsonl"
    dev_jsonl = "data/subtraction_sft/dev.jsonl"
    out_dir = "artifacts/sft/subtraction_clean"
    os.makedirs(out_dir, exist_ok=True)
    out_ckpt_path = os.path.join(out_dir, "best.pt")

    print("=" * 80)
    print(" MINI-R1 TASK 19C: CLEAN SUBTRACTION SFT TRAINING")
    print(f" Device: {device} | Base: {base_ckpt_path}")
    print(f" Effective Batch Size: {args.micro_batch_size * args.grad_accum_steps}")
    print(f" Peak LR: {args.peak_lr} | Min LR: {args.min_lr} | Epochs: {args.epochs}")
    print("=" * 80)

    # Hashes
    base_sha = compute_file_sha256(base_ckpt_path)
    tok_sha = compute_file_sha256(tok_path)
    print(f"Base Checkpoint SHA256: {base_sha}")
    print(f"Tokenizer SHA256:       {tok_sha}")

    tokenizer = Tokenizer(tok_path)
    train_dataset = MaskedSubtractionSFTDataset(train_jsonl, tokenizer, max_seq_len=256)
    dev_dataset = MaskedSubtractionSFTDataset(dev_jsonl, tokenizer, max_seq_len=256)

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

    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    total_optimizer_steps = args.epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_optimizer_steps)

    print(f"Total Steps: {total_optimizer_steps} | Warmup Steps: {warmup_steps} | Steps/Epoch: {steps_per_epoch}")

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
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                train_loss = accum_loss / args.grad_accum_steps
                accum_loss = 0.0

                if global_step % 25 == 0 or global_step == total_optimizer_steps:
                    elapsed = time.perf_counter() - t_start
                    tok_per_sec = total_tokens_seen / max(0.1, elapsed)
                    print(
                        f"Epoch {epoch:1d} | Step {global_step:4d}/{total_optimizer_steps} | "
                        f"Loss: {train_loss:6.4f} | LR: {curr_lr:8.2e} | "
                        f"GradNorm: {grad_norm:5.2f} | Speed: {tok_per_sec:5.0f} tok/s"
                    )

                if global_step % args.eval_interval == 0 or global_step == total_optimizer_steps:
                    dev_loss, dev_ppl = evaluate_dev_loss(model, dev_loader, device)
                    model.train()
                    is_best = dev_loss < best_dev_loss
                    if is_best:
                        best_dev_loss = dev_loss
                        best_dev_ppl = dev_ppl
                        best_step = global_step

                        torch.save({
                            "step": global_step,
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "model_config": model_cfg.to_dict(),
                            "dev_loss": dev_loss,
                            "dev_ppl": dev_ppl,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "base_checkpoint": base_ckpt_path,
                            "base_sha256": base_sha,
                        }, out_ckpt_path)

                    print(
                        f"  >>> [EVAL step {global_step}] Dev Loss: {dev_loss:.4f} | Dev PPL: {dev_ppl:.2f} | "
                        f"Best Dev Loss: {best_dev_loss:.4f} (step {best_step}) {'[SAVED BEST]' if is_best else ''}"
                    )

                    train_logs.append({
                        "step": global_step,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "dev_loss": dev_loss,
                        "dev_ppl": dev_ppl,
                        "lr": curr_lr,
                        "grad_norm": float(grad_norm),
                        "total_tokens_seen": total_tokens_seen,
                        "completion_tokens_seen": completion_tokens_seen,
                        "prompt_tokens_seen": prompt_tokens_seen,
                    })

                if global_step >= total_optimizer_steps:
                    break

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(" TASK 19C SFT TRAINING COMPLETE")
    print(f" Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f" Total Tokens Seen: {total_tokens_seen:,} (Completion: {completion_tokens_seen:,})")
    print(f" Best Step: {best_step} | Best Dev Loss: {best_dev_loss:.4f} | Best Dev PPL: {best_dev_ppl:.2f}")
    print(f" Saved Checkpoint: {out_ckpt_path} (SHA256: {compute_file_sha256(out_ckpt_path)})")
    print("=" * 80)

    # Save training logs
    os.makedirs("reports", exist_ok=True)
    with open("reports/task19c_training_log.json", "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "task": "Task 19C Clean Subtraction SFT Training",
                "base_checkpoint": base_ckpt_path,
                "base_sha256": base_sha,
                "output_checkpoint": out_ckpt_path,
                "output_sha256": compute_file_sha256(out_ckpt_path),
                "total_steps": total_optimizer_steps,
                "best_step": best_step,
                "best_dev_loss": best_dev_loss,
                "best_dev_ppl": best_dev_ppl,
                "total_time_seconds": total_time,
                "total_tokens_seen": total_tokens_seen,
                "completion_tokens_seen": completion_tokens_seen,
            },
            "logs": train_logs,
        }, f, indent=2)


if __name__ == "__main__":
    main()
