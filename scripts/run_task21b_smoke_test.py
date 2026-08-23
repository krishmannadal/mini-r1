"""
run_task21b_smoke_test.py - Task 21B Pre-Training Smoke Test.

Verifies:
1. Dataset verification and loading from data/gsm8k_compact_reasoning/
2. Batch collation with prompt-only loss masking (-100 on prompt)
3. Forward pass through MiniR1Transformer
4. Masked cross-entropy loss computation
5. Backward pass and gradient computation
6. Optimizer step with finite gradients (no NaNs, no Infs)
7. Checkpoint save, serialization, and round-trip weight verification
8. Executes a 10-step mini training loop.
"""

import os
import sys
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import MiniR1Config, MiniR1Transformer
from src.tokenizer import Tokenizer
from scripts.train_task21a_transfer_sft import GSM8KReasoningSFTDataset, collate_sft_batch


def run_smoke_test() -> None:
    print("=" * 70)
    print("MINI-R1 -- TASK 21B: PRE-TRAINING SMOKE TEST")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer_path = "artifacts/tokenizer/tokenizer.model"
    tokenizer = Tokenizer(tokenizer_path)
    print(f"Loaded tokenizer (Vocab: {tokenizer.vocab_size()})")

    train_jsonl = "data/gsm8k_compact_reasoning/train.jsonl"
    dev_jsonl = "data/gsm8k_compact_reasoning/dev.jsonl"
    assert os.path.exists(train_jsonl), f"Missing {train_jsonl}"
    assert os.path.exists(dev_jsonl), f"Missing {dev_jsonl}"

    print(f"\n[1/8] Initializing GSM8KReasoningSFTDataset...")
    dataset = GSM8KReasoningSFTDataset(train_jsonl, tokenizer, max_seq_len=512)
    dev_dataset = GSM8KReasoningSFTDataset(dev_jsonl, tokenizer, max_seq_len=512)
    print(f"Train dataset size: {len(dataset)} examples")
    print(f"Dev dataset size: {len(dev_dataset)} examples")

    print(f"\n[2/8] Testing DataLoader and Batch Collation...")
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda b: collate_sft_batch(b, tokenizer.pad_id),
    )
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    active_toks = batch["active_completion_tokens"]
    print(f"Batch input_ids shape: {input_ids.shape}")
    print(f"Batch labels shape: {labels.shape}")
    print(f"Active completion tokens in batch: {active_toks}")
    assert input_ids.shape == labels.shape, "Shape mismatch between input_ids and labels"
    assert (labels == -100).sum() > 0, "Loss masking failed: No -100 prompt labels found!"

    print(f"\n[3/8] Loading Model Architecture from Base Pretrained Checkpoint...")
    base_ckpt = "checkpoints/mini_r1_500m_pretrain/best.pt"
    assert os.path.exists(base_ckpt), f"Missing base checkpoint: {base_ckpt}"
    ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("model_config", ckpt.get("config", {}))
    config = MiniR1Config.from_dict(cfg_dict)
    model = MiniR1Transformer(config).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", {}))
    model.load_state_dict(state_dict)
    print(f"Loaded model ({sum(p.numel() for p in model.parameters()):,} parameters)")

    print(f"\n[4/8] Testing Forward Pass and Masked Loss Computation...")
    model.train()
    logits = model(input_ids)
    print(f"Logits shape: {logits.shape}")

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    print(f"Computed Masked Loss: {loss.item():.4f}")
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN or Inf!"

    print(f"\n[5/8] Testing Backward Pass and Gradient Computation...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    optimizer.zero_grad()
    loss.backward()

    max_grad = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(p.grad).any(), f"Inf gradient in {name}"
            g_norm = p.grad.norm().item()
            if g_norm > max_grad:
                max_grad = g_norm
    print(f"All gradients finite. Peak gradient norm: {max_grad:.4f}")

    print(f"\n[6/8] Testing Optimizer Step...")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    print("Optimizer step completed successfully.")

    print(f"\n[7/8] Testing Checkpoint Save and Weight Round-Trip...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_ckpt = os.path.join(tmpdir, "test_ckpt.pt")
        torch.save({"model": model.state_dict(), "config": config.to_dict()}, tmp_ckpt)
        loaded = torch.load(tmp_ckpt, map_location="cpu", weights_only=False)
        m2 = MiniR1Transformer(config)
        m2.load_state_dict(loaded["model"])
        for p1, p2 in zip(model.parameters(), m2.parameters()):
            assert torch.equal(p1.cpu(), p2), "Checkpoint weight mismatch!"
    print("Checkpoint save/load round-trip verified bitwise.")

    print(f"\n[8/8] Executing 10-Step Mini Training Run...")
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    model.train()
    loader_iter = iter(loader)
    for step in range(1, 11):
        try:
            b = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            b = next(loader_iter)

        in_ids = b["input_ids"].to(device)
        lbls = b["labels"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            out = model(in_ids)
            l = loss_fn(out.view(-1, out.size(-1)), lbls.view(-1))

        scaler.scale(l).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        print(f"  Step {step:02d}/10 | Loss: {l.item():.4f} | Active Tokens: {b['active_completion_tokens']}")

    print("\n" + "=" * 70)
    print("SMOKE TEST STATUS: PASSED ALL 8 CHECKS")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
