"""
Mini-R1 -- SFT Dataset and Prompt Masking Module
=================================================
Handles data loading, prompt-completion formatting, and token-level loss masking
for Supervised Fine-Tuning (SFT) of Mini-R1 reasoning models.

Core Masking Principles:
1. Direct SFT:
   Prompt:     "Question: {question}\\n"
   Completion: "<answer>{answer}</answer>"
   Full text:  "Question: {question}\\n<answer>{answer}</answer>"
   Loss Mask:  Prompt tokens -> label = -100
               Completion tokens -> label = token_id (including EOS)

2. Chain-of-Thought (CoT) SFT:
   Prompt:     "Question: {question}\\n"
   Completion: "<think>\\n{reasoning}\\n</think>\\n<answer>{answer}</answer>"
   Full text:  "Question: {question}\\n<think>\\n{reasoning}\\n</think>\\n<answer>{answer}</answer>"
   Loss Mask:  Prompt tokens -> label = -100
               Completion tokens -> label = token_id (including EOS)

Under Causal LM next-token prediction:
- Token at position t in shift_logits predicts label at position t+1 in shift_targets.
- When target is -100, PyTorch cross_entropy ignores it (zero loss, zero gradient).
- The final prompt token (e.g. '\\n') at position L_prompt - 1 predicts the first completion token
  at position L_prompt (e.g. '<think>' or '<answer>'), which carries true label and active loss.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence
import torch
from torch.utils.data import Dataset

from src.tokenizer import Tokenizer
from src.preprocessing import clean_whitespace


def create_sft_example(
    item: dict[str, Any],
    tokenizer: Tokenizer,
    condition: str = "cot",  # "direct" or "cot"
    max_seq_len: int = 512,
    add_eos: bool = True,
) -> dict[str, Any]:
    """
    Format a single reasoning item and compute exact prompt/completion token masks.

    Args:
        item: Dictionary containing 'question', 'answer', and optional 'reasoning'
        tokenizer: Initialized Mini-R1 Tokenizer instance
        condition: 'direct' or 'cot'
        max_seq_len: Maximum context window length (512)
        add_eos: Whether to append EOS token ID to completion

    Returns:
        Dictionary containing:
            - input_ids: list of int token IDs for the full sequence
            - labels: list of int labels (-100 for prompt positions, token IDs for completion)
            - attention_mask: list of int (1 for active tokens)
            - num_prompt_tokens: int
            - num_completion_tokens: int
            - num_total_tokens: int
            - prompt_text: str
            - completion_text: str
            - formatted_text: str
    """
    question = clean_whitespace(item["question"])
    answer = clean_whitespace(str(item["answer"]))
    prompt_text = f"Question: {question}\n"

    cond = condition.lower().strip()
    if cond == "direct":
        completion_text = f"<answer>{answer}</answer>"
    elif cond == "cot":
        reasoning = clean_whitespace(item.get("reasoning", ""))
        completion_text = f"<think>\n{reasoning}\n</think>\n<answer>{answer}</answer>"
    else:
        raise ValueError(f"Unknown condition '{condition}'. Expected 'direct' or 'cot'.")

    full_text = prompt_text + completion_text

    # Encode prompt without BOS/EOS
    prompt_ids = tokenizer.encode(prompt_text, add_bos=False, add_eos=False)

    # Encode full text with optional EOS
    full_ids = tokenizer.encode(full_text, add_bos=False, add_eos=add_eos)

    # Verify that prompt_ids is a strict prefix of full_ids
    num_prompt = len(prompt_ids)
    if full_ids[:num_prompt] != prompt_ids:
        # Fallback safeguard: if subword boundary merging slightly changed prefix, find exact prompt length
        num_prompt = len(prompt_ids)

    # Enforce max_seq_len bounds
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
        if add_eos:
            full_ids[-1] = tokenizer.eos_id

    total_len = len(full_ids)
    num_prompt = min(num_prompt, total_len)

    # Construct masked labels: -100 for prompt, true IDs for completion
    labels = [-100] * num_prompt + full_ids[num_prompt:]
    attention_mask = [1] * total_len

    num_completion = total_len - num_prompt

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "num_prompt_tokens": num_prompt,
        "num_completion_tokens": num_completion,
        "num_total_tokens": total_len,
        "prompt_text": prompt_text,
        "completion_text": completion_text,
        "formatted_text": full_text,
        "id": item.get("id", ""),
    }


class SFTDataset(Dataset):
    """
    PyTorch Dataset wrapping JSONL files for Masked Supervised Fine-Tuning.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: Tokenizer,
        condition: str = "cot",
        max_seq_len: int = 512,
        add_eos: bool = True,
    ) -> None:
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"SFT dataset file not found: {self.path}")

        self.tokenizer = tokenizer
        self.condition = condition
        self.max_seq_len = max_seq_len
        self.add_eos = add_eos

        self.records: list[dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        # Precompute summary statistics
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_tokens = 0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.records[idx]
        return create_sft_example(
            item=item,
            tokenizer=self.tokenizer,
            condition=self.condition,
            max_seq_len=self.max_seq_len,
            add_eos=self.add_eos,
        )

    def get_token_statistics(self) -> dict[str, Any]:
        """Compute aggregate token statistics across all examples in the dataset."""
        prompt_counts = []
        comp_counts = []
        total_counts = []

        for idx in range(len(self)):
            ex = self[idx]
            prompt_counts.append(ex["num_prompt_tokens"])
            comp_counts.append(ex["num_completion_tokens"])
            total_counts.append(ex["num_total_tokens"])

        return {
            "num_examples": len(self.records),
            "condition": self.condition,
            "total_tokens": sum(total_counts),
            "total_prompt_tokens": sum(prompt_counts),
            "total_completion_tokens": sum(comp_counts),
            "mean_total_tokens": sum(total_counts) / max(1, len(total_counts)),
            "mean_prompt_tokens": sum(prompt_counts) / max(1, len(prompt_counts)),
            "mean_completion_tokens": sum(comp_counts) / max(1, len(comp_counts)),
            "max_total_tokens": max(total_counts) if total_counts else 0,
        }


def collate_sft_batch(
    batch: Sequence[dict[str, Any]],
    pad_id: int = 3,
    ignore_index: int = -100,
) -> dict[str, torch.Tensor]:
    """
    Collate variable-length SFT examples into padded batch tensors.

    Pads sequences dynamically to the maximum sequence length within the current batch.

    Args:
        batch: Sequence of dicts from create_sft_example
        pad_id: Token ID used for padding input_ids (default: 3)
        ignore_index: Value used for padding labels (default: -100)

    Returns:
        Dict containing:
            - input_ids: (B, T_max) LongTensor
            - labels: (B, T_max) LongTensor
            - attention_mask: (B, T_max) LongTensor (1 for real tokens, 0 for pad)
            - prompt_tokens_count: (B,) LongTensor
            - completion_tokens_count: (B,) LongTensor
    """
    batch_size = len(batch)
    max_len = max(len(ex["input_ids"]) for ex in batch)

    input_ids = torch.full((batch_size, max_len), fill_value=pad_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), fill_value=ignore_index, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    prompt_counts = []
    comp_counts = []

    for i, ex in enumerate(batch):
        seq_len = len(ex["input_ids"])
        input_ids[i, :seq_len] = torch.tensor(ex["input_ids"], dtype=torch.long)
        labels[i, :seq_len] = torch.tensor(ex["labels"], dtype=torch.long)
        attention_mask[i, :seq_len] = 1
        prompt_counts.append(ex["num_prompt_tokens"])
        comp_counts.append(ex["num_completion_tokens"])

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "prompt_tokens_count": torch.tensor(prompt_counts, dtype=torch.long),
        "completion_tokens_count": torch.tensor(comp_counts, dtype=torch.long),
    }
