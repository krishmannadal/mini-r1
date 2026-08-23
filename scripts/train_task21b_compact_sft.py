"""
train_task21b_compact_sft.py - Task 21B: Compact Verified Reasoning Transfer SFT.

Trains:
- Model A: Initialized from Task 20 progressive arithmetic SFT (artifacts/sft/progressive_subtraction/best.pt)
- Model B: Initialized from Base Pretrained step 18k checkpoint (checkpoints/mini_r1_500m_pretrain/best.pt)

Hyperparameters (strictly identical between Model A and Model B):
- Training Data: data/gsm8k_compact_reasoning/train.jsonl (6,531 examples)
- Dev Data: data/gsm8k_compact_reasoning/dev.jsonl (470 examples)
- AdamW (peak LR=1e-4, min LR=1e-5, warmup=50 steps, weight_decay=0.01, max_grad_norm=1.0)
- Effective batch size: 32 (micro=8, accum=4)
- Optimizer steps: 650 steps
- Completion-only loss masking (-100 on prompt), max_seq_len=512, FP16 AMP
- Saves checkpoints to artifacts/sft/compact_transfer/arithmetic_init/ and base_init/
- Saves training logs to reports/task21b_training_logs.json
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


class CompactReasoningDataset(Dataset):
    """Dataset for compact verified GSM8K reasoning SFT with prompt token masking."""

    def __init__(self, jsonl_path: str, tokenizer: Tokenizer, max_seq_len: int = 512):
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
        if "prompt" in rec and "completion" in rec:
            prompt = rec["prompt"]
            completion = rec["completion"]
        else:
            q = rec["question"]
            r = rec.get("reasoning", "")
            a = rec["answer"]
            prompt = f"Question: {q}\n"
            completion = f"<think>\n{r}\n</think>\n<answer>{a}</answer>"

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


def collate_sft_batch(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    max_len = max(len(item["input_ids"]) for item in batch)
    padded_input_ids = []
    padded_labels = []
    total_active_completion = sum(item["active_completion_tokens"] for item in batch)

    for item in batch:
        in_seq = item["input_ids"]
        lbl_seq = item["labels"]
        pad_len = max_len - len(in_seq)

        if pad_len > 0:
            padded_in = torch.cat([in_seq, torch.full((pad_len,), pad_id, dtype=torch.long)])
            padded_lbl = torch.cat([lbl_seq, torch.full((pad_len,), -100, dtype=torch.long)])
        else:
            padded_in = in_seq
            padded_lbl = lbl_seq

        padded_input_ids.append(padded_in)
        padded_labels.append(padded_lbl)

    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels),
        "active_completion_tokens": total_active_completion,
    }


def get_cosine_lr(step: int, total_steps: int, peak_lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return min_lr + (peak_lr - min_lr) * (step / max(1, warmup_steps))
    if step > total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine_decay


@torch.no_grad()
def evaluate_dev_loss(
    model: MiniR1Transformer, dev_loader: DataLoader, device: str
) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    total_loss = 0.0
    total_active_tokens = 0

    for batch in dev_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=("cuda" in device)):
            logits = model(input_ids)
            loss_sum = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        active_tokens = (labels != -100).sum().item()
        total_loss += loss_sum.item()
        total_active_tokens += active_tokens

    avg_loss = total_loss / max(1, total_active_tokens)
    ppl = math.exp(min(20.0, avg_loss))
    return avg_loss, ppl


def train_single_model(
    model_name: str,
    init_ckpt_path: str,
    output_dir: str,
    train_jsonl: str,
    dev_jsonl: str,
    tok_path: str,
    device: str,
    max_steps: int = 650,
    peak_lr: float = 1e-4,
    min_lr: float = 1e-5,
    warmup_steps: int = 50,
    micro_batch_size: int = 8,
    grad_accum_steps: int = 4,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    eval_interval: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    print("\n" + "=" * 80)
    print(f" TRAINING {model_name.upper()}: Initialized from {init_ckpt_path}")
    print(f" Output Dir: {output_dir} | Target Steps: {max_steps} | Effective Batch: {micro_batch_size * grad_accum_steps}")
    print("=" * 80)

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.makedirs(output_dir, exist_ok=True)
    out_ckpt_path = os.path.join(output_dir, "best.pt")
    final_ckpt_path = os.path.join(output_dir, "final.pt")

    init_sha = compute_file_sha256(init_ckpt_path)
    tok_sha = compute_file_sha256(tok_path)
    print(f"Init Checkpoint SHA256: {init_sha}")
    print(f"Tokenizer SHA256:       {tok_sha}")

    tokenizer = Tokenizer(tok_path)
    train_dataset = CompactReasoningDataset(train_jsonl, tokenizer, max_seq_len=512)
    dev_dataset = CompactReasoningDataset(dev_jsonl, tokenizer, max_seq_len=512)

    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_sft_batch(b, pad_id=tokenizer.pad_id),
        generator=torch.Generator().manual_seed(seed),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=micro_batch_size * 2,
        shuffle=False,
        collate_fn=lambda b: collate_sft_batch(b, pad_id=tokenizer.pad_id),
    )

    # Load Model
    init_ckpt = torch.load(init_ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = init_ckpt.get("model_config", init_ckpt.get("config", {}))
    model_cfg = MiniR1Config.from_dict(cfg_dict)
    model = MiniR1Transformer(model_cfg).to(device)
    state_dict = init_ckpt.get("model_state_dict", init_ckpt.get("model", {}))
    model.load_state_dict(state_dict)
    print(f"Loaded Initialization Checkpoint (Params: {model.get_num_params():,})")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=("cuda" in device))
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    init_dev_loss, init_dev_ppl = evaluate_dev_loss(model, dev_loader, device)
    print(f"Initial Zero-Shot Dev Loss: {init_dev_loss:.4f} | PPL: {init_dev_ppl:.2f}")

    best_dev_loss = float("inf")
    best_step = 0
    history: list[dict[str, Any]] = []

    step = 0
    epoch = 0
    start_time = time.time()
    total_active_tokens_trained = 0

    running_train_loss = 0.0
    running_active_tokens = 0
    optimizer.zero_grad()

    while step < max_steps:
        epoch += 1
        model.train()
        for micro_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            active_tokens = batch["active_completion_tokens"]

            current_lr = get_cosine_lr(step, max_steps, peak_lr, min_lr, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            with torch.amp.autocast("cuda", enabled=("cuda" in device)):
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            running_train_loss += loss.item() * grad_accum_steps
            running_active_tokens += active_tokens
            total_active_tokens_trained += active_tokens

            if (micro_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1

                if step % 10 == 0 or step == 1 or step == max_steps:
                    elapsed = time.time() - start_time
                    tok_per_sec = total_active_tokens_trained / max(1.0, elapsed)
                    print(
                        f"Step {step:04d}/{max_steps} | Loss: {running_train_loss / 10:.4f} | "
                        f"LR: {current_lr:.2e} | Grad: {grad_norm:.3f} | Tok/s: {tok_per_sec:.0f}"
                    )
                    running_train_loss = 0.0

                if step % eval_interval == 0 or step == max_steps:
                    dev_loss, dev_ppl = evaluate_dev_loss(model, dev_loader, device)
                    print(f"--- [EVAL Step {step:04d}] Dev Loss: {dev_loss:.4f} | Dev PPL: {dev_ppl:.2f} ---")
                    
                    is_best = dev_loss < best_dev_loss
                    if is_best:
                        best_dev_loss = dev_loss
                        best_step = step
                        # Atomic save best.pt
                        tmp_ckpt = out_ckpt_path + ".tmp"
                        torch.save({
                            "step": step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "model_config": model_cfg.to_dict(),
                            "dev_loss": dev_loss,
                            "dev_ppl": dev_ppl,
                            "init_checkpoint_sha256": init_sha,
                            "tokenizer_sha256": tok_sha,
                            "model_name": model_name,
                        }, tmp_ckpt)
                        os.replace(tmp_ckpt, out_ckpt_path)
                        print(f"==> Saved NEW BEST Checkpoint to {out_ckpt_path} (Loss: {best_dev_loss:.4f})")

                    history.append({
                        "step": step,
                        "dev_loss": dev_loss,
                        "dev_ppl": dev_ppl,
                        "lr": current_lr,
                        "is_best": is_best,
                    })
                    model.train()

                if step >= max_steps:
                    break

    # Save final.pt
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_cfg.to_dict(),
        "dev_loss": dev_loss,
        "dev_ppl": dev_ppl,
        "init_checkpoint_sha256": init_sha,
        "tokenizer_sha256": tok_sha,
        "model_name": model_name,
    }, final_ckpt_path)
    print(f"Saved FINAL Checkpoint to {final_ckpt_path}")

    # Best SHA256
    best_sha = compute_file_sha256(out_ckpt_path)
    final_sha = compute_file_sha256(final_ckpt_path)

    # Save config and metadata
    cfg_path = os.path.join(output_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "init_checkpoint": init_ckpt_path,
            "init_checkpoint_sha256": init_sha,
            "best_checkpoint_sha256": best_sha,
            "final_checkpoint_sha256": final_sha,
            "max_steps": max_steps,
            "peak_lr": peak_lr,
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "micro_batch_size": micro_batch_size,
            "grad_accum_steps": grad_accum_steps,
            "effective_batch_size": micro_batch_size * grad_accum_steps,
            "best_step": best_step,
            "best_dev_loss": best_dev_loss,
            "best_dev_ppl": math.exp(min(20.0, best_dev_loss)),
            "total_active_tokens_trained": total_active_tokens_trained,
        }, f, indent=2)

    return {
        "model_name": model_name,
        "best_checkpoint": out_ckpt_path,
        "best_checkpoint_sha256": best_sha,
        "final_checkpoint_sha256": final_sha,
        "best_step": best_step,
        "best_dev_loss": best_dev_loss,
        "best_dev_ppl": math.exp(min(20.0, best_dev_loss)),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 21B Compact SFT Trainer")
    parser.add_argument("--steps", type=int, default=650, help="Total optimizer steps")
    parser.add_argument("--peak_lr", type=float, default=1e-4, help="Peak learning rate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_path = "artifacts/tokenizer/tokenizer.model"
    train_jsonl = "data/gsm8k_compact_reasoning/train.jsonl"
    dev_jsonl = "data/gsm8k_compact_reasoning/dev.jsonl"

    arith_init = "artifacts/sft/progressive_subtraction/best.pt"
    base_init = "checkpoints/mini_r1_500m_pretrain/best.pt"

    arith_out = "artifacts/sft/compact_transfer/arithmetic_init"
    base_out = "artifacts/sft/compact_transfer/base_init"

    # Train Model A (Arithmetic -> Compact GSM8K)
    res_a = train_single_model(
        model_name="Model A (Arithmetic -> Compact GSM8K)",
        init_ckpt_path=arith_init,
        output_dir=arith_out,
        train_jsonl=train_jsonl,
        dev_jsonl=dev_jsonl,
        tok_path=tok_path,
        device=device,
        max_steps=args.steps,
        peak_lr=args.peak_lr,
    )

    # Train Model B (Base -> Compact GSM8K)
    res_b = train_single_model(
        model_name="Model B (Base -> Compact GSM8K)",
        init_ckpt_path=base_init,
        output_dir=base_out,
        train_jsonl=train_jsonl,
        dev_jsonl=dev_jsonl,
        tok_path=tok_path,
        device=device,
        max_steps=args.steps,
        peak_lr=args.peak_lr,
    )

    # Save training logs report
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/task21b_training_logs.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "task": "Task 21B: Compact Verified Reasoning Transfer Training",
            "model_a": res_a,
            "model_b": res_b,
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("TASK 21B SFT TRAINING COMPLETE FOR BOTH MODELS")
    print(f"Model A Best Dev Loss: {res_a['best_dev_loss']:.4f} (PPL: {res_a['best_dev_ppl']:.2f})")
    print(f"Model B Best Dev Loss: {res_b['best_dev_loss']:.4f} (PPL: {res_b['best_dev_ppl']:.2f})")
    print(f"Logs saved to: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
