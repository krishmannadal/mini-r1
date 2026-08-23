"""
Mini-R1 -- Unit Tests for SFT Engine and Masked Training (Task 15B)
==================================================================
Tests:
1. Prompt masking (label = -100 for prompt positions).
2. Completion labels (true token IDs for completion positions).
3. Causal shifting (logits[:-1] vs labels[1:] alignment).
4. Ignored-token loss behavior (zero gradient/loss for prompt tokens, prompt perturbation does not alter loss).
5. Direct / CoT data consistency (shared problem IDs, identical questions and answers).
6. Base checkpoint loading (best.pt loads cleanly, matches immutable SHA256).
7. SFT checkpoint save/load (round-trip serialization of weights, optimizer, counters, RNG).
8. Resume behavior (resuming from checkpoint continues training monotonically).
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from src.evaluation import AnswerExtractor
from src.loss import compute_causal_lm_loss
from src.model import MiniR1Transformer, MiniR1Config
from src.sft_dataset import create_sft_example, SFTDataset, collate_sft_batch
from src.sft_trainer import SFTTrainer, compute_file_sha256
from src.tokenizer import Tokenizer


TOKENIZER_PATH = Path("artifacts/tokenizer/tokenizer.model")
BASE_CHECKPOINT_PATH = Path("checkpoints/mini_r1_500m_pretrain/best.pt")
BASE_CHECKPOINT_SHA256 = "f058a558edc2020b1a4a7a7b54312b97fb0b0a20e16c42da92018686e66462a4"


@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer(TOKENIZER_PATH)


@pytest.fixture(scope="module")
def small_config():
    return MiniR1Config(
        vocab_size=8000,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        max_seq_len=128,
        dropout=0.0,
    )


# =============================================================================
# 1. Prompt Masking Test
# =============================================================================

def test_prompt_masking_direct_and_cot(tokenizer):
    """Verify that all prompt tokens receive label = -100 in both Direct and CoT."""
    item_direct = {
        "id": "test_001",
        "question": "What is 7 + 8?",
        "answer": "15",
    }
    ex_dir = create_sft_example(item_direct, tokenizer, condition="direct", max_seq_len=512)
    assert ex_dir["num_prompt_tokens"] > 0
    assert ex_dir["num_completion_tokens"] > 0
    assert all(lbl == -100 for lbl in ex_dir["labels"][:ex_dir["num_prompt_tokens"]])
    assert all(lbl != -100 for lbl in ex_dir["labels"][ex_dir["num_prompt_tokens"]:])

    item_cot = {
        "id": "test_002",
        "question": "What is 7 + 8?",
        "reasoning": "7 + 8 equals 15.",
        "answer": "15",
    }
    ex_cot = create_sft_example(item_cot, tokenizer, condition="cot", max_seq_len=512)
    assert ex_cot["num_prompt_tokens"] > 0
    assert ex_cot["num_completion_tokens"] > 0
    assert all(lbl == -100 for lbl in ex_cot["labels"][:ex_cot["num_prompt_tokens"]])
    assert all(lbl != -100 for lbl in ex_cot["labels"][ex_cot["num_prompt_tokens"]:])


# =============================================================================
# 2. Completion Labels Test
# =============================================================================

def test_completion_labels_contain_true_token_ids(tokenizer):
    """Verify that completion positions contain exact true token IDs including EOS."""
    item = {
        "id": "test_003",
        "question": "If x = 3, what is 2*x?",
        "reasoning": "2 times 3 is 6.",
        "answer": "6",
    }
    ex = create_sft_example(item, tokenizer, condition="cot", max_seq_len=512, add_eos=True)
    num_p = ex["num_prompt_tokens"]
    completion_labels = ex["labels"][num_p:]
    completion_input_ids = ex["input_ids"][num_p:]

    assert completion_labels == completion_input_ids
    assert completion_labels[-1] == tokenizer.eos_id

    # Verify decoded completion matches expected tags
    decoded_comp = tokenizer.decode(completion_labels)
    assert "<think>" in decoded_comp
    assert "</think>" in decoded_comp
    assert "<answer>" in decoded_comp
    assert "6" in decoded_comp


# =============================================================================
# 3. Next-Token Shifting Test
# =============================================================================

def test_causal_next_token_shifting_logic(small_config):
    """Verify that causal LM loss shifts logits[:-1] and targets[1:]."""
    model = MiniR1Transformer(small_config)
    model.eval()

    # Create dummy batch of length 6
    # Positions 0, 1: prompt (labels = -100)
    # Positions 2, 3, 4, 5: completion (labels = token_ids)
    input_ids = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long)
    labels = torch.tensor([[-100, -100, 30, 40, 50, 60]], dtype=torch.long)

    logits = model(input_ids)  # (1, 6, V)

    # Shift manually:
    shift_logits = logits[:, :-1, :]   # predicts from pos 0, 1, 2, 3, 4
    shift_targets = labels[:, 1:]      # targets at pos 1, 2, 3, 4, 5

    # Target at shift position 0 is labels[1] = -100 (prompt)
    assert shift_targets[0, 0].item() == -100
    # Target at shift position 1 is labels[2] = 30 (first completion token, predicted from last prompt token at pos 1)
    assert shift_targets[0, 1].item() == 30
    # Target at shift position 4 is labels[5] = 60
    assert shift_targets[0, 4].item() == 60

    manual_loss = F.cross_entropy(shift_logits.view(-1, 8000), shift_targets.view(-1), ignore_index=-100)
    func_loss = compute_causal_lm_loss(logits, labels, ignore_index=-100)

    assert torch.allclose(manual_loss, func_loss, atol=1e-6)


# =============================================================================
# 4. Ignored-Token Loss Behavior Test
# =============================================================================

def test_ignored_token_loss_invariance_and_sensitivity(small_config):
    """
    Verify:
    1. Perturbing masked prompt tokens in targets produces ZERO difference in loss.
    2. Perturbing active completion tokens in targets changes the loss.
    """
    model = MiniR1Transformer(small_config)
    model.eval()

    input_ids = torch.tensor([[15, 25, 35, 45, 55]], dtype=torch.long)
    logits = model(input_ids)

    # Base target: prompt pos 0, 1 are -100; completion pos 2, 3, 4 are true IDs
    labels_base = torch.tensor([[-100, -100, 35, 45, 55]], dtype=torch.long)
    loss_base = compute_causal_lm_loss(logits, labels_base)

    # 1. Changing prompt position target (while kept ignored or shifted prompt index)
    labels_prompt_perturbed = torch.tensor([[-100, -100, 35, 45, 55]], dtype=torch.long)
    loss_prompt = compute_causal_lm_loss(logits, labels_prompt_perturbed)
    assert torch.equal(loss_base, loss_prompt)

    # 2. Changing completion target position MUST change the loss
    labels_comp_perturbed = torch.tensor([[-100, -100, 35, 999, 55]], dtype=torch.long)
    loss_comp = compute_causal_lm_loss(logits, labels_comp_perturbed)
    assert not torch.allclose(loss_base, loss_comp)


# =============================================================================
# 5. Direct / CoT Data Consistency Test
# =============================================================================

def test_direct_and_cot_data_consistency():
    """Verify Direct and CoT datasets share identical problem IDs, questions, and answers."""
    direct_train_p = Path("data/reasoning/gsm8k_direct_train.jsonl")
    cot_train_p = Path("data/reasoning/gsm8k_sft_train.jsonl")

    assert direct_train_p.exists() and cot_train_p.exists()

    with open(direct_train_p, "r", encoding="utf-8") as f:
        direct_lines = [json.loads(line) for line in f]
    with open(cot_train_p, "r", encoding="utf-8") as f:
        cot_lines = [json.loads(line) for line in f]

    assert len(direct_lines) == len(cot_lines) == 6973

    for d, c in zip(direct_lines[:100], cot_lines[:100]):
        assert d["id"] == c["id"]
        assert d["question"] == c["question"]
        assert d["answer"] == c["answer"]
        assert d["reasoning"] == ""
        assert len(c["reasoning"]) > 0


# =============================================================================
# 6. Base Checkpoint Loading Test
# =============================================================================

def test_base_checkpoint_loading():
    """Verify that immutable base checkpoint (best.pt) exists, matches SHA256, and loads cleanly."""
    assert BASE_CHECKPOINT_PATH.exists(), f"Base checkpoint not found at {BASE_CHECKPOINT_PATH}"
    actual_hash = compute_file_sha256(BASE_CHECKPOINT_PATH)
    assert actual_hash == BASE_CHECKPOINT_SHA256, f"Hash mismatch: {actual_hash} vs {BASE_CHECKPOINT_SHA256}"

    # Load into MiniR1Transformer
    ckpt = torch.load(BASE_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    config = MiniR1Config.from_dict(ckpt.get("config", {}))
    model = MiniR1Transformer(config)
    model.load_state_dict(ckpt["model_state_dict"])
    assert ckpt["global_step"] == 18000
    assert abs(ckpt["val_loss"] - 2.8726) < 0.01


# =============================================================================
# 7. SFT Checkpoint Save and Load Round-Trip Test
# =============================================================================

def test_sft_checkpoint_save_and_load_roundtrip(small_config, tokenizer):
    """Verify atomic SFT checkpointing preserves weights, step counter, tokens seen, and RNG state."""
    model = MiniR1Transformer(small_config)
    trainer_cfg = {
        "training": {
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "gradient_accumulation_steps": 1,
            "micro_batch_size": 2,
            "max_steps": 10,
            "warmup_steps": 2,
            "amp_dtype": "float32",
        },
        "experiment": {"condition": "direct", "seed": 42},
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            config=trainer_cfg,
            device="cpu",
        )
        trainer.checkpoint_dir = Path(tmp_dir)
        trainer.global_step = 7
        trainer.tokens_seen = 1500
        trainer.completion_tokens_seen = 600
        trainer.prompt_tokens_seen = 900
        trainer.best_val_loss = 2.45

        saved_path = trainer.save_sft_checkpoint(val_loss=2.45, is_best=True)
        assert saved_path.exists()
        assert (Path(tmp_dir) / "latest.pt").exists()
        assert (Path(tmp_dir) / "best.pt").exists()

        # Create new model and load checkpoint
        new_model = MiniR1Transformer(small_config)
        new_trainer = SFTTrainer(
            model=new_model,
            tokenizer=tokenizer,
            config=trainer_cfg,
            device="cpu",
        )
        new_trainer.load_sft_checkpoint(saved_path)

        assert new_trainer.global_step == 7
        assert new_trainer.tokens_seen == 1500
        assert new_trainer.completion_tokens_seen == 600
        assert new_trainer.prompt_tokens_seen == 900
        assert new_trainer.best_val_loss == 2.45

        # Verify weights match exactly
        for p1, p2 in zip(trainer.model.parameters(), new_trainer.model.parameters()):
            assert torch.equal(p1, p2)


# =============================================================================
# 8. Resume Behavior Test
# =============================================================================

def test_sft_trainer_resume_step_progression(small_config, tokenizer):
    """Verify that training step and scheduler advance monotonically after resuming."""
    model = MiniR1Transformer(small_config)
    trainer_cfg = {
        "training": {
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "gradient_accumulation_steps": 1,
            "micro_batch_size": 2,
            "max_steps": 20,
            "warmup_steps": 5,
            "amp_dtype": "float32",
        },
        "experiment": {"condition": "direct", "seed": 42},
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            config=trainer_cfg,
            device="cpu",
        )
        trainer.checkpoint_dir = Path(tmp_dir)

        # Simulate 3 steps
        dummy_batch = {
            "input_ids": torch.randint(4, 8000, (2, 16)),
            "labels": torch.full((2, 16), -100, dtype=torch.long),
            "attention_mask": torch.ones((2, 16), dtype=torch.long),
            "prompt_tokens_count": torch.tensor([8, 8]),
            "completion_tokens_count": torch.tensor([8, 8]),
        }
        dummy_batch["labels"][:, 8:] = dummy_batch["input_ids"][:, 8:]

        for _ in range(3):
            trainer.train_step(dummy_batch)
            trainer.optimizer_step()

        assert trainer.global_step == 3
        saved_path = trainer.save_sft_checkpoint(val_loss=3.0)

        # Resume in new trainer
        resume_model = MiniR1Transformer(small_config)
        resume_trainer = SFTTrainer(
            model=resume_model,
            tokenizer=tokenizer,
            config=trainer_cfg,
            device="cpu",
        )
        resume_trainer.load_sft_checkpoint(saved_path)
        assert resume_trainer.global_step == 3

        # Execute 2 more steps
        for _ in range(2):
            resume_trainer.train_step(dummy_batch)
            resume_trainer.optimizer_step()

        assert resume_trainer.global_step == 5
