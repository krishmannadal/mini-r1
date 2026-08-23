"""
Mini-R1 -- Model & Training Memory Estimation Module
===================================================
Provides analytical memory breakdown calculation for Transformer models without
needing high-level abstractions.

Memory Components:
    1. Model Weights:      P * bytes_per_elem
    2. Gradients:          P * bytes_per_elem
    3. Optimizer States:   2 * P * bytes_per_elem (AdamW stores 1st and 2nd moments)
    4. Static Training:    Weights + Gradients + Optimizer
    5. Activations:        Retained tensors for backprop per layer and output head
    6. Runtime Overhead:   ~10% PyTorch/CUDA workspace & fragmentation buffer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from src.model import MiniR1Config


@dataclass
class LayerParameterBreakdown:
    """Per-layer parameter counts for Transformer decoder block."""
    ln1_params: int
    q_proj_params: int
    k_proj_params: int
    v_proj_params: int
    out_proj_params: int
    ln2_params: int
    ffn_fc1_params: int
    ffn_fc2_params: int

    @property
    def attention_params(self) -> int:
        return self.q_proj_params + self.k_proj_params + self.v_proj_params + self.out_proj_params

    @property
    def ffn_params(self) -> int:
        return self.ffn_fc1_params + self.ffn_fc2_params

    @property
    def total_layer_params(self) -> int:
        return (
            self.ln1_params
            + self.attention_params
            + self.ln2_params
            + self.ffn_params
        )


@dataclass
class ParameterBreakdown:
    """Full model parameter breakdown."""
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    max_seq_len: int
    tie_word_embeddings: bool

    # Non-layer parameters
    tok_emb_params: int
    pos_emb_params: int
    ln_f_params: int
    lm_head_params: int

    # Per-layer breakdown
    layer_breakdown: LayerParameterBreakdown

    @property
    def all_layers_params(self) -> int:
        return self.layer_breakdown.total_layer_params * self.n_layers

    @property
    def non_layer_params(self) -> int:
        return (
            self.tok_emb_params
            + self.pos_emb_params
            + self.ln_f_params
            + self.lm_head_params
        )

    @property
    def total_params(self) -> int:
        return self.all_layers_params + self.non_layer_params

    @property
    def non_embedding_params(self) -> int:
        return self.total_params - self.tok_emb_params - self.pos_emb_params


@dataclass
class MemoryEstimation:
    """Full memory estimation report for a given training configuration."""
    batch_size: int
    seq_len: int
    dtype_bytes: int
    total_params: int

    # Component bytes
    weights_bytes: int
    gradients_bytes: int
    optimizer_bytes: int
    static_bytes: int
    activations_bytes: int
    overhead_bytes: int
    total_training_bytes: int

    # VRAM comparison
    vram_budget_gb: float
    fits_in_vram: bool

    @property
    def weights_mb(self) -> float:
        return self.weights_bytes / (1024 ** 2)

    @property
    def gradients_mb(self) -> float:
        return self.gradients_bytes / (1024 ** 2)

    @property
    def optimizer_mb(self) -> float:
        return self.optimizer_bytes / (1024 ** 2)

    @property
    def static_mb(self) -> float:
        return self.static_bytes / (1024 ** 2)

    @property
    def activations_mb(self) -> float:
        return self.activations_bytes / (1024 ** 2)

    @property
    def overhead_mb(self) -> float:
        return self.overhead_bytes / (1024 ** 2)

    @property
    def total_training_mb(self) -> float:
        return self.total_training_bytes / (1024 ** 2)

    @property
    def total_training_gb(self) -> float:
        return self.total_training_bytes / (1024 ** 3)


def calculate_parameter_breakdown(config: MiniR1Config) -> ParameterBreakdown:
    """
    Compute exact analytical parameter counts for the given MiniR1Config.
    """
    d = config.d_model
    d_ff = config.d_ff
    V = config.vocab_size
    T = config.max_seq_len
    bias = config.bias

    # LayerNorm parameters: weight (d) + bias (d) = 2d
    ln_params = 2 * d

    # Attention projections: Q, K, V, Out
    linear_bias_attn = d if bias else 0
    q_proj = d * d + linear_bias_attn
    k_proj = d * d + linear_bias_attn
    v_proj = d * d + linear_bias_attn
    out_proj = d * d + linear_bias_attn

    # FFN projections: fc1 (d -> d_ff), fc2 (d_ff -> d)
    linear_bias_fc1 = d_ff if bias else 0
    linear_bias_fc2 = d if bias else 0
    ffn_fc1 = d * d_ff + linear_bias_fc1
    ffn_fc2 = d_ff * d + linear_bias_fc2

    layer_breakdown = LayerParameterBreakdown(
        ln1_params=ln_params,
        q_proj_params=q_proj,
        k_proj_params=k_proj,
        v_proj_params=v_proj,
        out_proj_params=out_proj,
        ln2_params=ln_params,
        ffn_fc1_params=ffn_fc1,
        ffn_fc2_params=ffn_fc2,
    )

    tok_emb = V * d
    pos_emb = T * d
    ln_f = 2 * d
    # If tied, lm_head has 0 unique additional parameters; else V * d
    lm_head = 0 if config.tie_word_embeddings else (V * d)

    return ParameterBreakdown(
        vocab_size=V,
        d_model=d,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=d_ff,
        max_seq_len=T,
        tie_word_embeddings=config.tie_word_embeddings,
        tok_emb_params=tok_emb,
        pos_emb_params=pos_emb,
        ln_f_params=ln_f,
        lm_head_params=lm_head,
        layer_breakdown=layer_breakdown,
    )


def estimate_training_memory(
    config: MiniR1Config,
    batch_size: int = 8,
    seq_len: int | None = None,
    dtype_bytes: int = 4,  # FP32 = 4 bytes, FP16/BF16 = 2 bytes
    optimizer_states_per_param: int = 2,  # AdamW (first & second moments)
    overhead_ratio: float = 0.10,  # 10% workspace & CUDA allocator overhead
    vram_budget_gb: float = 6.0,
) -> MemoryEstimation:
    """
    Analytically estimate training memory for weights, gradients, optimizer, activations,
    and runtime overhead.
    """
    param_info = calculate_parameter_breakdown(config)
    P = param_info.total_params
    T = seq_len if seq_len is not None else config.max_seq_len
    B = batch_size
    d = config.d_model
    H = config.n_heads
    L = config.n_layers
    V = config.vocab_size
    d_ff = config.d_ff

    # 1. Static model components
    weights_bytes = P * dtype_bytes
    gradients_bytes = P * dtype_bytes
    # Optimizer states are stored in FP32 even in mixed precision
    optimizer_bytes = P * optimizer_states_per_param * 4
    static_bytes = weights_bytes + gradients_bytes + optimizer_bytes

    # 2. Activation elements retained for backward pass
    # Per layer:
    #   - LN1 residual input: B * T * d
    #   - Q, K, V after projection: 3 * B * T * d
    #   - Attention matrix (softmax): B * H * T * T
    #   - Attention context: B * T * d
    #   - LN2 residual input: B * T * d
    #   - FFN hidden (post-fc1 for GELU backprop): B * T * d_ff
    #   - FFN output: B * T * d
    per_layer_activation_elems = (
        (B * T * d)          # LN1 input
        + (3 * B * T * d)    # Q, K, V
        + (B * H * T * T)    # Attn matrix
        + (B * T * d)        # Attn context
        + (B * T * d)        # LN2 input
        + (B * T * d_ff)     # FFN intermediate
        + (B * T * d)        # FFN output
    )
    all_layers_activation_elems = per_layer_activation_elems * L

    # Non-layer activations:
    #   - Embeddings output: B * T * d
    #   - Final LN input: B * T * d
    #   - Logits: B * T * V
    non_layer_activation_elems = (B * T * d) + (B * T * d) + (B * T * V)

    total_activation_elems = all_layers_activation_elems + non_layer_activation_elems
    activations_bytes = total_activation_elems * dtype_bytes

    # 3. Overhead and Total
    subtotal_bytes = static_bytes + activations_bytes
    overhead_bytes = int(subtotal_bytes * overhead_ratio)
    total_training_bytes = subtotal_bytes + overhead_bytes

    total_training_gb = total_training_bytes / (1024 ** 3)
    fits_in_vram = total_training_gb <= vram_budget_gb

    return MemoryEstimation(
        batch_size=B,
        seq_len=T,
        dtype_bytes=dtype_bytes,
        total_params=P,
        weights_bytes=weights_bytes,
        gradients_bytes=gradients_bytes,
        optimizer_bytes=optimizer_bytes,
        static_bytes=static_bytes,
        activations_bytes=activations_bytes,
        overhead_bytes=overhead_bytes,
        total_training_bytes=total_training_bytes,
        vram_budget_gb=vram_budget_gb,
        fits_in_vram=fits_in_vram,
    )
