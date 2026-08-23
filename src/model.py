"""
Mini-R1 -- Decoder-Only Transformer Language Model
==================================================
The core causal Transformer architecture implemented from scratch.

Architecture Summary:
    - Vocabulary size       : Configurable (default: 8000)
    - Embedding dimension   : 128 (d_model)
    - Transformer layers    : 2 (n_layers)
    - Attention heads       : 4 (n_heads, d_k = 32)
    - Feed-forward dimension: 512 (d_ff)
    - Max sequence length   : 32 (max_seq_len)

Forward Pass Progression:
    input_ids: (B, T)
        -> Embeddings (tok + pos) : (B, T, 128)
        -> Layer 1 Block          : (B, T, 128)
        -> Layer 2 Block          : (B, T, 128)
        -> Final LayerNorm        : (B, T, 128)
        -> Language Modeling Head : (B, T, vocab_size) [Logits]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple
import torch
import torch.nn as nn

from src.embeddings import TransformerEmbedding
from src.transformer_block import TransformerBlock, BlockOutputs


from pathlib import Path
import yaml


@dataclass
class MiniR1Config:
    """Configuration hyperparameters for Mini-R1 decoder Transformer."""
    vocab_size: int = 8000
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.0
    bias: bool = False
    tie_word_embeddings: bool = True

    @property
    def d_k(self) -> int:
        """Attention head dimension."""
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        return self.d_model // self.n_heads

    @classmethod
    def from_dict(cls, data: dict) -> MiniR1Config:
        """Create a MiniR1Config instance from a dictionary."""
        model_dict = data.get("model", data)
        return cls(
            vocab_size=int(model_dict.get("vocab_size", 8000)),
            d_model=int(model_dict.get("d_model", 512)),
            n_layers=int(model_dict.get("n_layers", 12)),
            n_heads=int(model_dict.get("n_heads", 8)),
            d_ff=int(model_dict.get("d_ff", 2048)),
            max_seq_len=int(model_dict.get("max_seq_len", 512)),
            dropout=float(model_dict.get("dropout", 0.0)),
            bias=bool(model_dict.get("bias", False)),
            tie_word_embeddings=bool(model_dict.get("tie_word_embeddings", True)),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> MiniR1Config:
        """Load configuration from a YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Serialize configuration to a dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "bias": self.bias,
            "tie_word_embeddings": self.tie_word_embeddings,
        }


class ModelForwardOutputs(NamedTuple):
    """Container holding logits and intermediate tensor traces."""
    logits: torch.Tensor
    embedding_out: torch.Tensor
    block_outputs: list[BlockOutputs]


class MiniR1Transformer(nn.Module):
    """
    Decoder-only Causal Transformer Language Model.
    """

    def __init__(self, config: MiniR1Config | None = None) -> None:
        super().__init__()
        self.config = config or MiniR1Config()

        # 1. Embeddings: Token + Learned Positional
        self.embeddings = TransformerEmbedding(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.dropout,
        )

        # 2. Transformer Decoder Blocks (Pre-LN Stack)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                d_ff=self.config.d_ff,
                max_seq_len=self.config.max_seq_len,
                dropout=self.config.dropout,
                bias=self.config.bias,
            )
            for _ in range(self.config.n_layers)
        ])

        # 3. Final Layer Normalization
        self.ln_f = nn.LayerNorm(self.config.d_model)

        # 4. Language Modeling Head (Linear projection from d_model to vocab_size)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

        # Weight tying: tie token embedding matrix and lm_head projection weights
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embeddings.tok_emb.weight

        # Initialize weights with standard normal (std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with Gaussian distribution scaled by 0.02."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor | ModelForwardOutputs:
        """
        Complete forward pass of the decoder-only Transformer.

        Args:
            input_ids: LongTensor of token IDs with shape (B, T)
            return_intermediates: If True, exposes all internal tensor traces for inspection.

        Returns:
            Logits tensor of shape (B, T, vocab_size) or ModelForwardOutputs namedtuple.
        """
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}"
            )

        # 1. Embeddings -> (B, T, d_model)
        emb_out = self.embeddings(input_ids)
        x = emb_out

        # 2. Transformer Decoder Blocks -> (B, T, d_model)
        block_traces: list[BlockOutputs] = []
        for block in self.blocks:
            if return_intermediates:
                block_res: BlockOutputs = block(x, return_intermediates=True)
                x = block_res.output
                block_traces.append(block_res)
            else:
                x = block(x, return_intermediates=False)

        # 3. Final LayerNorm -> (B, T, d_model)
        x = self.ln_f(x)

        # 4. Language Modeling Head -> (B, T, vocab_size)
        logits = self.lm_head(x)

        if return_intermediates:
            return ModelForwardOutputs(
                logits=logits,
                embedding_out=emb_out,
                block_outputs=block_traces,
            )

        return logits

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Return the total number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embeddings.pos_emb.weight.numel()
            if not self.config.tie_word_embeddings:
                n_params -= self.embeddings.tok_emb.weight.numel()
        return n_params
