"""
train_task19d_mixed_sft.py - SFT Training Engine for Task 19D.

Identical training infrastructure to Task 19C but on mixed 2+3-digit subtraction data.
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


class MaskedSFTDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: Tokenizer, max_seq_len: int = 256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: list[dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.examples[idx]
        prompt_toks = self.tokenizer.encode(rec["prompt"], add_bos=True, add_eos=False)
        comp_toks = self.tokenizer.encode(rec["completion"], add_bos=False, add_eos=True)
        full_seq = (prompt_toks + comp_toks)[:self.max_seq_len]
        input_ids = full_seq[:-1]
        target_ids = full_seq[1:]
        num_prompt_targets = max(0, len(prompt_toks) - 1)
        labels = list(target_ids)
        for i in range(min(num_prompt_targets, len(labels))):
            labels[i] = -100
        active = sum(1 for x in labels if x != -100)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_tokens": len(prompt_toks),
            "completion_tokens": len(comp_toks),
            "active_completion_tokens": active,
        }


def collate_batch(batch: list[dict], pad_id: int) -> dict:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    labels = []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
        labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "total_prompt_tokens": sum(x["prompt_tokens"] for x in batch),
        "total_completion_tokens": sum(x["completion_tokens"] for x in batch),
        "active_completion_tokens": sum(x["active_completion_tokens"] for x in batch),
    }


def get_lr(step: int, warmup: int, total: int, peak: float, minimum: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


@torch.inference_mode()
def eval_dev(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    for batch in loader:
        ids = batch["input_ids"].to(device)
        labs = batch["labels"].to(device)
        with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16):
            logits = model(ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labs.view(-1))
        total_loss += loss.item()
        total_tokens += (labs != -100).sum().item()
    avg = total_loss / max(1, total_tokens)
    return avg, math.exp(min(avg, 20.0))


def main():
    parser = argparse.ArgumentParser()
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_path = "checkpoints/mini_r1_500m_pretrain/best.pt"
    tok_path = "artifacts/tokenizer/tokenizer.model"
    train_path = "data/mixed_subtraction_sft/train.jsonl"
    dev_path = "data/mixed_subtraction_sft/dev.jsonl"
    out_dir = "artifacts/sft/mixed_subtraction"
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best.pt")

    print("=" * 80)
    print(" MINI-R1 TASK 19D: MIXED 2+3-DIGIT SUBTRACTION SFT")
    print(f" Device: {device} | Effective Batch: {args.micro_batch_size * args.grad_accum_steps}")
    print("=" * 80)

    base_sha = compute_file_sha256(base_path)
    tok_sha = compute_file_sha256(tok_path)
    print(f"Base SHA256: {base_sha}")
    print(f"Tok SHA256:  {tok_sha}")

    tokenizer = Tokenizer(tok_path)
    train_ds = MaskedSFTDataset(train_path, tokenizer)
    dev_ds = MaskedSFTDataset(dev_path, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.micro_batch_size, shuffle=True,
                              collate_fn=lambda b: collate_batch(b, tokenizer.pad_id),
                              generator=torch.Generator().manual_seed(args.seed))
    dev_loader = DataLoader(dev_ds, batch_size=args.micro_batch_size * 2, shuffle=False,
                            collate_fn=lambda b: collate_batch(b, tokenizer.pad_id))

    base_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    cfg = MiniR1Config.from_dict(base_ckpt.get("model_config", base_ckpt.get("config", {})))
    model = MiniR1Transformer(cfg).to(device)
    model.load_state_dict(base_ckpt["model_state_dict"])
    print(f"Loaded Base (Step {base_ckpt.get('step', 18000)}, Params: {model.get_num_params():,})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled="cuda" in device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    total_steps = args.epochs * steps_per_epoch
    warmup = int(0.05 * total_steps)
    print(f"Total Steps: {total_steps} | Warmup: {warmup} | Steps/Epoch: {steps_per_epoch}")

    init_loss, init_ppl = eval_dev(model, dev_loader, device)
    print(f"Initial Dev Loss: {init_loss:.4f} | PPL: {init_ppl:.2f}")

    best_loss = float("inf")
    best_step = 0
    logs = []
    step = 0
    total_toks = 0
    comp_toks = 0
    t0 = time.perf_counter()
    epoch = 0

    while step < total_steps:
        epoch += 1
        model.train()
        accum = 0.0
        micro = 0

        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            labs = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16):
                logits = model(ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labs.view(-1))
                loss_s = loss / args.grad_accum_steps

            scaler.scale(loss_s).backward()
            accum += loss.item()
            micro += 1
            total_toks += ids.numel()
            comp_toks += batch["total_completion_tokens"]

            if micro % args.grad_accum_steps == 0:
                step += 1
                lr = get_lr(step, warmup, total_steps, args.peak_lr, args.min_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                scaler.unscale_(optimizer)
                gn = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                train_loss = accum / args.grad_accum_steps
                accum = 0.0

                if step % 25 == 0 or step == total_steps:
                    elapsed = time.perf_counter() - t0
                    print(f"Ep {epoch} | Step {step:4d}/{total_steps} | Loss: {train_loss:.4f} | "
                          f"LR: {lr:.2e} | GN: {gn:.2f} | {total_toks/elapsed:.0f} tok/s")

                if step % args.eval_interval == 0 or step == total_steps:
                    dl, dp = eval_dev(model, dev_loader, device)
                    model.train()
                    is_best = dl < best_loss
                    if is_best:
                        best_loss = dl
                        best_step = step
                        torch.save({
                            "step": step, "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "model_config": cfg.to_dict(),
                            "dev_loss": dl, "dev_ppl": dp,
                            "base_sha256": base_sha,
                        }, best_path)
                    print(f"  >>> [EVAL {step}] Dev: {dl:.4f} PPL: {dp:.2f} | "
                          f"Best: {best_loss:.4f} (step {best_step}) {'[SAVED]' if is_best else ''}")
                    logs.append({"step": step, "train_loss": train_loss, "dev_loss": dl, "dev_ppl": dp, "lr": lr})

                if step >= total_steps:
                    break

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*80}")
    print(f" TASK 19D TRAINING COMPLETE")
    print(f" Time: {elapsed:.1f}s | Best Step: {best_step} | Best Dev Loss: {best_loss:.4f}")
    print(f" Checkpoint: {best_path} (SHA256: {compute_file_sha256(best_path)})")
    print(f"{'='*80}")

    with open("reports/task19d_training_log.json", "w") as f:
        json.dump({"metadata": {"task": "19D", "best_step": best_step, "best_loss": best_loss,
                                "time_s": elapsed, "total_toks": total_toks, "comp_toks": comp_toks},
                   "logs": logs}, f, indent=2)


if __name__ == "__main__":
    main()
