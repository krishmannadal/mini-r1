"""
test_arithmetic_sft_pipeline.py - Tests for SFT data loading, collating, loss masking, and forward/backward.
"""

import os
import torch
from torch.utils.data import DataLoader

from scripts.train_arithmetic_sft import (
    MaskedArithmeticSFTDataset,
    collate_sft_batch,
)
from src.model import MiniR1Config, MiniR1Transformer
from src.tokenizer import Tokenizer


def test_sft_dataset_masking():
    tok_path = "artifacts/tokenizer/tokenizer.model"
    tokenizer = Tokenizer(tok_path)
    train_jsonl = "data/arithmetic_sft/compact_dev.jsonl"

    dataset = MaskedArithmeticSFTDataset(train_jsonl, tokenizer, max_seq_len=256)
    assert len(dataset) == 1200

    item = dataset[0]
    assert "input_ids" in item
    assert "labels" in item
    assert item["input_ids"].shape == item["labels"].shape

    # First few label tokens must be masked (-100)
    labels = item["labels"].tolist()
    assert -100 in labels
    assert labels[0] == -100


def test_sft_collate_and_forward_backward():
    tok_path = "artifacts/tokenizer/tokenizer.model"
    tokenizer = Tokenizer(tok_path)
    train_jsonl = "data/arithmetic_sft/compact_dev.jsonl"

    dataset = MaskedArithmeticSFTDataset(train_jsonl, tokenizer, max_seq_len=256)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda b: collate_sft_batch(b, pad_id=tokenizer.pad_id),
    )

    batch = next(iter(loader))
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    assert input_ids.shape[0] == 4
    assert labels.shape == input_ids.shape

    # Small test model
    cfg = MiniR1Config(
        vocab_size=tokenizer.vocab_size(),
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        max_seq_len=256,
    )
    model = MiniR1Transformer(cfg)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    logits = model(input_ids)
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.all(torch.isfinite(p.grad))
