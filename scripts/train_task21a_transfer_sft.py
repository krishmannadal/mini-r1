"""
train_task21a_transfer_sft.py - Task 21A: Arithmetic -> GSM8K Transfer SFT Trainer.

Trains:
- Model A: Initialized from Task 20 progressive arithmetic checkpoint (artifacts/sft/progressive_subtraction/best.pt)
- Model B: Initialized from Base Pretrained step 18k checkpoint (checkpoints/mini_r1_500m_pretrain/best.pt)

Hyperparameters strictly identical:
- AdamW (peak LR=1e-4, min LR=1e-5, warmup=50 steps, weight_decay=0.01, max_grad_norm=1.0)
- Effective batch size: 32 (micro=8, accum=4)
- Optimizer steps: 650 steps (~3.0 epochs over 6,973 GSM8K examples)
- Completion-only loss masking, max_seq_len=512, FP16 AMP.
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


class GSM8KReasoningSFTDataset(Dataset):
    """Dataset for GSM8K CoT SFT with prompt token masking."""

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
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    labels = []

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len
        input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
        labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
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

    init_sha = compute_file_sha256(init_ckpt_path)
    tok_sha = compute_file_sha256(tok_path)
    print(f"Init Checkpoint SHA256: {init_sha}")
    print(f"Tokenizer SHA256:       {tok_sha}")

    tokenizer = Tokenizer(tok_path)
    train_dataset = GSM8KReasoningSFTDataset(train_jsonl, tokenizer, max_seq_len=512)
    dev_dataset = GSM8KReasoningSFTDataset(dev_jsonl, tokenizer, max_seq_len=512)

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
    model.load_state_dict(init_ckpt["model_state_dict"])
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
    best_dev_ppl = float("inf")
    best_step = 0

    train_logs: list[dict[str, Any]] = []
    global_step = 0
    total_tokens_seen = 0
    completion_tokens_seen = 0
    prompt_tokens_seen = 0

    t_start = time.perf_counter()
    epoch = 0

    while global_step < max_steps:
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
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()
            accum_loss += loss.item()
            micro_step += 1

            total_tokens_seen += input_ids.numel()
            prompt_tokens_seen += batch["total_prompt_tokens"]
            completion_tokens_seen += batch["total_completion_tokens"]

            if micro_step % grad_accum_steps == 0:
                global_step += 1
                curr_lr = get_lr(global_step, warmup_steps, max_steps, peak_lr, min_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = curr_lr

                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                train_loss = accum_loss / grad_accum_steps
                accum_loss = 0.0

                if global_step % 25 == 0 or global_step == max_steps:
                    elapsed = time.perf_counter() - t_start
                    print(
                        f"[{model_name}] Ep {epoch:1d} | Step {global_step:3d}/{max_steps} | "
                        f"Loss: {train_loss:6.4f} | LR: {curr_lr:8.2e} | "
                        f"GN: {grad_norm:5.2f} | Speed: {total_tokens_seen/elapsed:5.0f} tok/s"
                    )

                if global_step % eval_interval == 0 or global_step == max_steps:
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
                            "init_checkpoint": init_ckpt_path,
                            "init_sha256": init_sha,
                        }, out_ckpt_path)

                    print(
                        f"  >>> [{model_name} EVAL step {global_step}] Dev Loss: {dev_loss:.4f} | Dev PPL: {dev_ppl:.2f} | "
                        f"Best: {best_dev_loss:.4f} (step {best_step}) {'[SAVED BEST]' if is_best else ''}"
                    )

                    train_logs.append({
                        "step": global_step,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "dev_loss": dev_loss,
                        "dev_ppl": dev_ppl,
                        "lr": curr_lr,
                        "grad_norm": float(grad_norm),
                    })

                if global_step >= max_steps:
                    break

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f" {model_name.upper()} TRAINING COMPLETE")
    print(f" Time: {elapsed:.1f}s ({elapsed/60:.1f} min) | Best Step: {best_step} | Best Dev Loss: {best_dev_loss:.4f} | Dev PPL: {best_dev_ppl:.2f}")
    print(f" Saved: {out_ckpt_path} (SHA256: {compute_file_sha256(out_ckpt_path)})")
    print("=" * 80)

    return {
        "model_name": model_name,
        "init_checkpoint": init_ckpt_path,
        "init_sha256": init_sha,
        "checkpoint": out_ckpt_path,
        "checkpoint_sha256": compute_file_sha256(out_ckpt_path),
        "best_step": best_step,
        "best_dev_loss": best_dev_loss,
        "best_dev_ppl": best_dev_ppl,
        "total_time_seconds": elapsed,
        "total_tokens_seen": total_tokens_seen,
        "completion_tokens_seen": completion_tokens_seen,
        "logs": train_logs,
    }


def main():
    parser = argparse.ArgumentParser(description="Task 21A Transfer SFT Trainer")
    parser.add_argument("--model", type=str, choices=["both", "model_a", "model_b"], default="both")
    parser.add_argument("--steps", type=int, default=650)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_path = "artifacts/tokenizer/tokenizer.model"
    train_jsonl = "data/reasoning/gsm8k_sft_train.jsonl"
    dev_jsonl = "data/reasoning/gsm8k_sft_dev.jsonl"

    model_a_init = "artifacts/sft/progressive_subtraction/best.pt"
    model_b_init = "checkpoints/mini_r1_500m_pretrain/best.pt"

    all_logs = {}

    if args.model in ["both", "model_a"]:
        res_a = train_single_model(
            model_name="Model A (Arithmetic -> GSM8K)",
            init_ckpt_path=model_a_init,
            output_dir="artifacts/sft/arithmetic_transfer",
            train_jsonl=train_jsonl,
            dev_jsonl=dev_jsonl,
            tok_path=tok_path,
            device=device,
            max_steps=args.steps,
        )
        all_logs["model_a"] = res_a

    if args.model in ["both", "model_b"]:
        res_b = train_single_model(
            model_name="Model B (Base -> GSM8K)",
            init_ckpt_path=model_b_init,
            output_dir="artifacts/sft/base_transfer",
            train_jsonl=train_jsonl,
            dev_jsonl=dev_jsonl,
            tok_path=tok_path,
            device=device,
            max_steps=args.steps,
        )
        all_logs["model_b"] = res_b

    os.makedirs("reports", exist_ok=True)
    with open("reports/task21a_training_logs.json", "w", encoding="utf-8") as f:
        json.dump(all_logs, f, indent=2)


if __name__ == "__main__":
    main()
