"""
Mini-R1 -- Metrics Tracking Module
==================================
Tracks loss, token throughput, perplexity, and training progression metrics.
"""

from __future__ import annotations

import math
import time
from typing import NamedTuple


class MetricsSummary(NamedTuple):
    """Summary of training or evaluation metrics."""
    avg_loss: float
    perplexity: float
    total_tokens: int
    tokens_per_sec: float
    elapsed_sec: float


class MetricsTracker:
    """
    Cumulative metric tracker for training and evaluation epochs.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators and restart timer."""
        self.total_loss = 0.0
        self.total_tokens = 0
        self.step_count = 0
        self.start_time = time.perf_counter()

    def update(self, loss: float, num_tokens: int) -> None:
        """
        Record a single step's loss and token count.

        Args:
            loss: Scalar cross-entropy loss for the step
            num_tokens: Number of non-pad tokens processed in this step
        """
        self.total_loss += loss
        self.total_tokens += num_tokens
        self.step_count += 1

    def compute(self) -> MetricsSummary:
        """
        Compute aggregate statistics over the accumulated steps.
        """
        elapsed = max(1e-5, time.perf_counter() - self.start_time)
        avg_loss = (self.total_loss / self.step_count) if self.step_count > 0 else 0.0
        try:
            ppl = math.exp(min(20.0, avg_loss))
        except OverflowError:
            ppl = float("inf")

        tokens_per_sec = self.total_tokens / elapsed

        return MetricsSummary(
            avg_loss=avg_loss,
            perplexity=ppl,
            total_tokens=self.total_tokens,
            tokens_per_sec=tokens_per_sec,
            elapsed_sec=elapsed,
        )
