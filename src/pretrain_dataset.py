"""
Mini-R1 -- Sharded Pretraining Dataset Module
=============================================
Provides lazy, memory-efficient reading of local binary token shards (uint16)
with zero RAM bloat, deterministic sequence indexing, cursor tracking, and
exact checkpoint restoration.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ShardMetadata:
    shard_id: int
    filename: str
    num_sequences: int
    num_tokens: int
    file_size_bytes: int
    sha256: str
    md5: str


@dataclass
class DatasetPositionState:
    current_shard_idx: int = 0
    current_seq_in_shard: int = 0
    global_seq_idx: int = 0
    tokens_seen: int = 0
    manifest_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_shard_idx": self.current_shard_idx,
            "current_seq_in_shard": self.current_seq_in_shard,
            "global_seq_idx": self.global_seq_idx,
            "tokens_seen": self.tokens_seen,
            "manifest_hash": self.manifest_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetPositionState:
        return cls(
            current_shard_idx=data.get("current_shard_idx", 0),
            current_seq_in_shard=data.get("current_seq_in_shard", 0),
            global_seq_idx=data.get("global_seq_idx", 0),
            tokens_seen=data.get("tokens_seen", 0),
            manifest_hash=data.get("manifest_hash", ""),
        )


class ShardedPretrainDataset(Dataset):
    """
    PyTorch Dataset reading packed 512-token sequences directly from local binary shards.
    Keeps only the active shard in memory (or memmap) to prevent RAM saturation.
    """

    def __init__(
        self,
        manifest_path: Path | str,
        seq_len: int = 512,
        use_memmap: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Shard manifest not found at: {self.manifest_path}")

        self.seq_len = seq_len
        self.use_memmap = use_memmap
        self.shards_dir = self.manifest_path.parent / "shards"

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.tokenizer_md5 = self.manifest.get("tokenizer_md5", "")
        self.manifest_hash = self.manifest.get("manifest_hash", "")
        self.total_tokens_target = self.manifest.get("total_tokens", 0)

        # Parse shards list
        self.shards_meta: list[ShardMetadata] = []
        self.shard_cumulative_seqs: list[int] = [0]

        for s in self.manifest.get("shards", []):
            meta = ShardMetadata(
                shard_id=s["shard_id"],
                filename=s["filename"],
                num_sequences=s["num_sequences"],
                num_tokens=s["num_tokens"],
                file_size_bytes=s.get("file_size_bytes", s["num_sequences"] * seq_len * 2),
                sha256=s.get("sha256", ""),
                md5=s.get("md5", ""),
            )
            self.shards_meta.append(meta)
            self.shard_cumulative_seqs.append(
                self.shard_cumulative_seqs[-1] + meta.num_sequences
            )

        self.total_sequences = self.shard_cumulative_seqs[-1] if self.shard_cumulative_seqs else 0
        self.total_tokens = self.total_sequences * self.seq_len

        # Active shard cache (single shard in memory)
        self._cached_shard_idx: int = -1
        self._cached_shard_data: np.ndarray | None = None
        self.start_sequence_offset: int = 0

        # Cursor state
        self.state = DatasetPositionState(
            current_shard_idx=0,
            current_seq_in_shard=0,
            global_seq_idx=0,
            tokens_seen=0,
            manifest_hash=self.manifest_hash,
        )

    def __len__(self) -> int:
        return max(0, self.total_sequences - self.start_sequence_offset)

    def _load_shard(self, shard_idx: int) -> np.ndarray:
        if shard_idx == self._cached_shard_idx and self._cached_shard_data is not None:
            return self._cached_shard_data

        if shard_idx < 0 or shard_idx >= len(self.shards_meta):
            raise IndexError(f"Shard index {shard_idx} out of range [0, {len(self.shards_meta)-1}]")

        meta = self.shards_meta[shard_idx]
        shard_file = self.shards_dir / meta.filename
        if not shard_file.exists():
            # Check relative to manifest parent if not in shards/
            shard_file = self.manifest_path.parent / meta.filename
        if not shard_file.exists():
            raise FileNotFoundError(f"Shard file missing: {shard_file}")

        num_seqs = meta.num_sequences
        expected_shape = (num_seqs, self.seq_len)

        if self.use_memmap:
            data = np.memmap(
                shard_file,
                dtype=np.uint16,
                mode="r",
                shape=expected_shape,
            )
        else:
            flat = np.fromfile(shard_file, dtype=np.uint16)
            data = flat.reshape(expected_shape)

        self._cached_shard_idx = shard_idx
        self._cached_shard_data = data
        return data

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Sequence index {idx} out of bounds for dataset of length {len(self)}")

        # Map logical index to absolute continuous sequence index in shard array
        actual_idx = (self.start_sequence_offset + idx) % self.total_sequences

        # Binary search for shard containing actual_idx
        shard_idx = np.searchsorted(self.shard_cumulative_seqs, actual_idx, side="right") - 1
        seq_in_shard = actual_idx - self.shard_cumulative_seqs[shard_idx]

        shard_data = self._load_shard(shard_idx)
        raw_tokens = shard_data[seq_in_shard]

        # Update cursor state
        self.state.current_shard_idx = int(shard_idx)
        self.state.current_seq_in_shard = int(seq_in_shard)
        self.state.global_seq_idx = int(actual_idx + 1)
        self.state.tokens_seen = (actual_idx + 1) * self.seq_len

        # Convert uint16 to PyTorch LongTensor (int64) for causal LM
        return torch.from_numpy(raw_tokens.astype(np.int64))

    def get_state(self) -> dict[str, Any]:
        """Return serialized state dict for checkpointing."""
        return self.state.to_dict()

    def load_state(self, state_dict: dict[str, Any]) -> None:
        """Restore exact shard and sequence position from checkpoint."""
        self.state = DatasetPositionState.from_dict(state_dict)
        self.start_sequence_offset = self.state.global_seq_idx
        if 0 <= self.state.current_shard_idx < len(self.shards_meta):
            self._load_shard(self.state.current_shard_idx)

    def get_progress(self) -> dict[str, Any]:
        """Return token-budget progress statistics."""
        pct = (self.state.tokens_seen / max(1, self.total_tokens_target)) * 100.0
        return {
            "current_shard": self.state.current_shard_idx,
            "total_shards": len(self.shards_meta),
            "current_sequence": self.state.global_seq_idx,
            "total_sequences": self.total_sequences,
            "tokens_seen": self.state.tokens_seen,
            "total_tokens_target": self.total_tokens_target,
            "percent_complete": pct,
        }
