"""
Mini-R1 -- Learning Rate Scheduler Module
=========================================
Implements a clean, transparent Cosine Decay learning rate scheduler with linear warmup.
Provides complete state serialization (state_dict / load_state_dict) for checkpointing.

Mathematical formulation:
    For step t:
        If t < warmup_steps:
            lr(t) = base_lr * (t / warmup_steps)
        Else:
            progress = (t - warmup_steps) / (max_steps - warmup_steps)
            lr(t) = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
"""

from __future__ import annotations

import math
from typing import Any
from torch.optim import Optimizer


class CosineWarmupScheduler:
    """
    Cosine learning rate scheduler with linear warmup.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        base_lr: float | None = None,
        min_lr: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(0, warmup_steps)
        self.max_steps = max(1, max_steps)
        self.min_lr = min_lr

        # If base_lr not provided, deduce from optimizer's first param group
        if base_lr is None:
            self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        else:
            self.base_lrs = [base_lr for _ in optimizer.param_groups]

        self.current_step = 0
        self._last_lrs = list(self.base_lrs)

        # Set initial LR for step 0
        self.step(step=0)

    def get_lr_factor(self, step: int) -> float:
        """Compute the learning rate scaling factor for a given step."""
        if step < self.warmup_steps:
            # Linear warmup from 0 to 1
            if self.warmup_steps == 0:
                return 1.0
            return float(step) / float(self.warmup_steps)

        if step >= self.max_steps:
            return 0.0

        # Cosine decay from 1.0 down to min_lr / base_lr
        progress = float(step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine_factor

    def step(self, step: int | None = None) -> list[float]:
        """
        Advance the scheduler by one step (or set to specific step).
        Updates optimizer param_groups learning rates.
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        factor = self.get_lr_factor(self.current_step)
        new_lrs = []

        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            # Scale from base_lr down to min_lr
            lr = self.min_lr + (base_lr - self.min_lr) * factor
            group["lr"] = lr
            new_lrs.append(lr)

        self._last_lrs = new_lrs
        return new_lrs

    def get_last_lr(self) -> list[float]:
        """Return the most recently computed learning rates."""
        return self._last_lrs

    def state_dict(self) -> dict[str, Any]:
        """Serialize scheduler state."""
        return {
            "current_step": self.current_step,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "min_lr": self.min_lr,
            "base_lrs": self.base_lrs,
            "_last_lrs": self._last_lrs,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore scheduler state."""
        self.current_step = state_dict["current_step"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.max_steps = state_dict["max_steps"]
        self.min_lr = state_dict["min_lr"]
        self.base_lrs = state_dict["base_lrs"]
        self._last_lrs = state_dict.get("_last_lrs", list(self.base_lrs))

        # Re-apply restored learning rate to optimizer
        self.step(step=self.current_step)
