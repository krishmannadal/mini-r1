"""
Mini-R1 -- PyTorch Dataset & DataLoader Test Suite
==================================================
Tests verifying:
1. ReasoningTorchDataset tokenization, BOS/EOS inclusion, and truncation.
2. CausalCollateFn dynamic padding and target -100 masking.
3. DataLoader batch iteration shapes.
"""

from pathlib import Path
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.tokenizer import Tokenizer
from src.torch_dataset import ReasoningTorchDataset
from src.dataloader import CausalCollateFn, create_dataloader


@pytest.fixture(scope="session")
def tokenizer() -> Tokenizer:
    tok_path = PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.model"
    return Tokenizer(tok_path)


class TestTorchDatasetAndCollate:
    """Verify PyTorch dataset and collation operations."""

    def test_dataset_item_encoding(self, tokenizer: Tokenizer):
        """Dataset item is a 1D LongTensor starting with BOS (1) and ending with EOS (2)."""
        records = [
            {"formatted_text": "Question: 1+1?\n<think>1+1=2</think>\n<answer>2</answer>"},
            {"formatted_text": "Question: 2*3?\n<think>2*3=6</think>\n<answer>6</answer>"},
        ]
        ds = ReasoningTorchDataset(records, tokenizer=tokenizer, max_seq_len=32)

        assert len(ds) == 2
        item0 = ds[0]
        assert isinstance(item0, torch.Tensor)
        assert item0.ndim == 1
        assert item0[0].item() == tokenizer.bos_id
        assert item0[-1].item() == tokenizer.eos_id
        assert len(item0) <= 32

    def test_collate_fn_padding_and_masking(self):
        """Collate function pads shorter sequences with pad_id (3) and masks targets with -100."""
        seq1 = torch.tensor([1, 10, 20, 2], dtype=torch.long)  # len 4
        seq2 = torch.tensor([1, 5, 2], dtype=torch.long)        # len 3

        collate = CausalCollateFn(pad_id=3, ignore_index=-100)
        batch = collate([seq1, seq2])

        assert batch.input_ids.shape == (2, 4)
        assert batch.targets.shape == (2, 4)
        assert batch.attention_mask.shape == (2, 4)

        # seq2 was padded by 1 at index 3
        assert batch.input_ids[1, 3].item() == 3
        assert batch.targets[1, 3].item() == -100
        assert batch.attention_mask[1, 3].item() == 0

        # seq1 real tokens are not masked
        assert batch.targets[0, 3].item() == 2
        assert batch.attention_mask[0, 3].item() == 1

    def test_dataloader_iteration(self, tokenizer: Tokenizer):
        """DataLoader yields batches with matching 2D tensor dimensions."""
        records = [
            {"formatted_text": f"Question: Q{i}?\n<think>R{i}</think>\n<answer>{i}</answer>"}
            for i in range(10)
        ]
        ds = ReasoningTorchDataset(records, tokenizer=tokenizer, max_seq_len=32)
        loader = create_dataloader(ds, batch_size=4, shuffle=False, pad_id=tokenizer.pad_id)

        batches = list(loader)
        assert len(batches) == 3  # 4 + 4 + 2
        assert batches[0].input_ids.shape[0] == 4
        assert batches[2].input_ids.shape[0] == 2
