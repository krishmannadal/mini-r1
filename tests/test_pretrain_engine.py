"""
Mini-R1 -- Tests for Production Pretraining Engine (Task 12)
============================================================
Tests binary shard creation, lazy loading, cursor state saving/restoration,
checkpoint integration with dataset position, and network-isolated execution.
"""

import json
from pathlib import Path
import socket
import numpy as np
import pytest
import torch

from src.model import MiniR1Transformer, MiniR1Config
from src.pretrain_dataset import ShardedPretrainDataset, DatasetPositionState
from src.dataloader import create_dataloader
from src.checkpoint import save_checkpoint, load_checkpoint, CheckpointManager
from src.trainer import Trainer
from src.optimizer import create_adamw_optimizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def dummy_shard_manifest(tmp_path: Path):
    """Create two small dummy binary uint16 shards and a manifest."""
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    seq_len = 512
    num_seqs_1 = 10
    num_seqs_2 = 15

    # Create shard 1: sequence i has token values i
    data1 = np.full((num_seqs_1, seq_len), 100, dtype=np.uint16)
    for i in range(num_seqs_1):
        data1[i, :] = (i + 1) % 8000
    file1 = shards_dir / "train_shard_00000.bin"
    with open(file1, "wb") as f:
        f.write(data1.tobytes())

    # Create shard 2: sequence i has token values 100 + i
    data2 = np.full((num_seqs_2, seq_len), 200, dtype=np.uint16)
    for i in range(num_seqs_2):
        data2[i, :] = (100 + i) % 8000
    file2 = shards_dir / "train_shard_00001.bin"
    with open(file2, "wb") as f:
        f.write(data2.tobytes())

    manifest_data = {
        "manifest_name": "Test Pretrain Manifest",
        "split": "train",
        "tokenizer_md5": "dummy_md5",
        "vocab_size": 8000,
        "sequence_length": seq_len,
        "total_shards": 2,
        "total_sequences": num_seqs_1 + num_seqs_2,
        "total_tokens": (num_seqs_1 + num_seqs_2) * seq_len,
        "shards": [
            {
                "shard_id": 0,
                "filename": "train_shard_00000.bin",
                "num_sequences": num_seqs_1,
                "num_tokens": num_seqs_1 * seq_len,
                "file_size_bytes": len(data1.tobytes()),
            },
            {
                "shard_id": 1,
                "filename": "train_shard_00001.bin",
                "num_sequences": num_seqs_2,
                "num_tokens": num_seqs_2 * seq_len,
                "file_size_bytes": len(data2.tobytes()),
            },
        ],
    }

    manifest_path = tmp_path / "train_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_path


class TestShardedPretrainDataset:
    def test_lazy_loading_and_shard_transition(self, dummy_shard_manifest):
        dataset = ShardedPretrainDataset(dummy_shard_manifest, seq_len=512)
        assert len(dataset) == 25  # 10 + 15
        assert dataset.total_tokens == 25 * 512

        # Index within shard 0
        item0 = dataset[0]
        assert isinstance(item0, torch.Tensor)
        assert item0.shape == (512,)
        assert item0.dtype == torch.int64
        assert item0[0].item() == 1

        # Index within shard 0 last element
        item9 = dataset[9]
        assert item9[0].item() == 10
        assert dataset.state.current_shard_idx == 0
        assert dataset.state.current_seq_in_shard == 9

        # Index into shard 1 (transition)
        item10 = dataset[10]
        assert item10[0].item() == 100
        assert dataset.state.current_shard_idx == 1
        assert dataset.state.current_seq_in_shard == 0

        # Last item in dataset
        item24 = dataset[24]
        assert item24[0].item() == 114
        assert dataset.state.current_shard_idx == 1
        assert dataset.state.current_seq_in_shard == 14

    def test_cursor_state_save_and_load(self, dummy_shard_manifest):
        dataset = ShardedPretrainDataset(dummy_shard_manifest, seq_len=512)
        item12 = dataset[12]  # shard 1, seq 2 (absolute index 12)
        state = dataset.get_state()

        assert state["current_shard_idx"] == 1
        assert state["current_seq_in_shard"] == 2
        assert state["global_seq_idx"] == 13

        # New instance restore
        dataset2 = ShardedPretrainDataset(dummy_shard_manifest, seq_len=512)
        dataset2.load_state(state)
        assert len(dataset2) == 25 - 13
        # First item of resumed dataset should be item 13
        item_resumed_0 = dataset2[0]
        assert item_resumed_0[0].item() == 103  # 100 + (13 - 10)


class TestCheckpointDatasetResumability:
    def test_checkpoint_saves_and_restores_dataset_position(self, tmp_path, dummy_shard_manifest):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        cfg = MiniR1Config(vocab_size=8000, max_seq_len=512, d_model=128, n_layers=2, n_heads=2, d_ff=256)
        model = MiniR1Transformer(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        dataset = ShardedPretrainDataset(dummy_shard_manifest, seq_len=512)
        _ = dataset[15]  # shard 1, seq 5
        ds_state = dataset.get_state()

        ckpt_file = save_checkpoint(
            checkpoint_path=ckpt_dir / "test.pt",
            model=model,
            optimizer=opt,
            global_step=50,
            dataset_state=ds_state,
            tokens_seen=15 * 512,
        )

        loaded = load_checkpoint(ckpt_file)
        assert "dataset_state" in loaded
        assert loaded["dataset_state"]["current_shard_idx"] == 1
        assert loaded["dataset_state"]["current_seq_in_shard"] == 5
        assert loaded["tokens_seen"] == 15 * 512


class TestNetworkIsolation:
    def test_dataloader_reads_without_network(self, dummy_shard_manifest, monkeypatch):
        """Mock socket connection to ensure zero network calls occur during dataloading."""
        def guard_socket(*args, **kwargs):
            raise RuntimeError("Network socket access attempted during offline training!")

        monkeypatch.setattr(socket, "socket", guard_socket)

        dataset = ShardedPretrainDataset(dummy_shard_manifest, seq_len=512)
        loader = create_dataloader(dataset, batch_size=4, shuffle=False)

        batches = list(loader)
        assert len(batches) == 7  # ceil(25 / 4)
        assert batches[0].input_ids.shape == (4, 512)
