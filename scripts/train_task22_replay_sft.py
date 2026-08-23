"""
train_task22_replay_sft.py - Task 22: Operator-Matched Replay SFT with Parameter Isolation.

Thin driver around the Task 21B SFT trainer. Every piece of core training
machinery is IMPORTED from scripts/train_task21b_compact_sft.py (dataset,
collate, cosine schedule, dev-loss evaluator) so that no existing file is
modified and the optimisation path is identical to Task 21B.

What this script adds, and nothing else:
  1. A 50/50 strictly interleaved mixture of compact GSM8K + replay arithmetic.
  2. Parameter isolation (freeze tok_emb / pos_emb / blocks 0-5; train blocks
     6-11 + ln_f), driven by regex patterns from the YAML config.
  3. Per-source dev-loss logging.

Step budget (650), effective batch (32), peak LR (1e-4) and seed (42) are read
from configs/task22_replay_sft.yaml and match Task 21B exactly.

Usage:
  python scripts/train_task22_replay_sft.py --config configs/task22_replay_sft.yaml
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
for _p in (PROJECT_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- reuse the Task 21B trainer verbatim; nothing in it is modified ---------
from train_task21b_compact_sft import (  # noqa: E402
    CompactReasoningDataset,
    collate_sft_batch,
    compute_file_sha256,
    evaluate_dev_loss,
    get_cosine_lr,
)
from src.model import MiniR1Config, MiniR1Transformer  # noqa: E402
from src.tokenizer import Tokenizer  # noqa: E402


# --------------------------------------------------------------------------- #
# 50/50 interleaved mixture
# --------------------------------------------------------------------------- #
class InterleavedMixtureDataset(Dataset):
    """
    Strict alternation between two CompactReasoningDataset sources.

    Index 2k   -> source A (compact GSM8K)
    Index 2k+1 -> source B (replay arithmetic)

    Consumed with shuffle=False, so with micro_batch_size=8 every micro-batch
    holds exactly 4 examples from each source: a genuine 50/50 interleave rather
    than a 50/50 expectation. Each source is independently permuted per epoch by
    reshuffle(epoch); the shorter source cycles.
    """

    def __init__(self, ds_a: Dataset, ds_b: Dataset, seed: int, epoch_pairs: int | None = None):
        self.ds_a = ds_a
        self.ds_b = ds_b
        self.seed = seed
        self.epoch_pairs = epoch_pairs if epoch_pairs is not None else min(len(ds_a), len(ds_b))
        self.order_a: list[int] = []
        self.order_b: list[int] = []
        self.reshuffle(0)

    def _order(self, n_src: int, rng: random.Random) -> list[int]:
        out: list[int] = []
        while len(out) < self.epoch_pairs:
            perm = list(range(n_src))
            rng.shuffle(perm)
            out.extend(perm)
        return out[: self.epoch_pairs]

    def reshuffle(self, epoch: int) -> None:
        self.order_a = self._order(len(self.ds_a), random.Random(self.seed + 7919 * epoch))
        self.order_b = self._order(len(self.ds_b), random.Random(self.seed + 104729 * epoch + 1))

    def __len__(self) -> int:
        return self.epoch_pairs * 2

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        k = idx // 2
        if idx % 2 == 0:
            return self.ds_a[self.order_a[k]]
        return self.ds_b[self.order_b[k]]


class RatioScheduledMixtureDataset(Dataset):
    """
    Task 23: deterministic global-index replay scheduling.

        is_replay = (idx % period == 0)

    period=10 -> exactly 10% replay slots; period=4 -> exactly 25%.
    Consecutive replay hits alternate between the replay sources, so a
    two-source replay block splits exactly 50/50.

    No fractional per-batch quotas: the ratio is exact over the epoch because
    the epoch length is forced to a whole number of periods. Each source is
    independently permuted per epoch; shorter sources cycle deterministically.
    """

    def __init__(self, primary, replay_sources, period, seed):
        self.primary = primary
        self.replay_sources = list(replay_sources)
        self.period = int(period)
        self.seed = seed

        raw = math.ceil(len(primary) * self.period / (self.period - 1))
        self.epoch_len = int(math.ceil(raw / self.period) * self.period)
        self.n_replay = self.epoch_len // self.period
        self.n_primary = self.epoch_len - self.n_replay

        self.order_primary = []
        self.orders_replay = []
        self.reshuffle(0)

    @staticmethod
    def _order(n_src, need, rng):
        out = []
        while len(out) < need:
            perm = list(range(n_src))
            rng.shuffle(perm)
            out.extend(perm)
        return out[:need]

    def reshuffle(self, epoch):
        self.order_primary = self._order(
            len(self.primary), self.n_primary, random.Random(self.seed + 7919 * epoch)
        )
        self.orders_replay = []
        k = len(self.replay_sources)
        for j, ds in enumerate(self.replay_sources):
            need = len(range(j, self.n_replay, k))
            self.orders_replay.append(
                self._order(len(ds), max(1, need),
                            random.Random(self.seed + 104729 * epoch + 31 * j + 1))
            )

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        if idx % self.period == 0:
            hit = idx // self.period
            j = hit % len(self.replay_sources)
            pos = hit // len(self.replay_sources)
            order = self.orders_replay[j]
            return self.replay_sources[j][order[pos % len(order)]]
        pos = idx - (idx // self.period) - 1
        return self.primary[self.order_primary[pos % len(self.order_primary)]]


# --------------------------------------------------------------------------- #
# Parameter isolation
# --------------------------------------------------------------------------- #
def apply_parameter_isolation(model: nn.Module, patterns: list[str]) -> dict[str, Any]:
    """
    Freeze every parameter whose name does not match one of `patterns`.
    Returns an auditable record of exactly what was frozen and what was trained.
    """
    compiled = [re.compile(p) for p in patterns]
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    n_trainable = 0
    n_frozen = 0

    for name, param in model.named_parameters():
        if any(c.search(name) for c in compiled):
            param.requires_grad_(True)
            trainable_names.append(name)
            n_trainable += param.numel()
        else:
            param.requires_grad_(False)
            frozen_names.append(name)
            n_frozen += param.numel()

    def summarise(names: list[str]) -> list[str]:
        return sorted({re.sub(r"\.\d+\.", ".{i}.", n) for n in names})

    return {
        "patterns": patterns,
        "trainable_param_names": trainable_names,
        "frozen_param_names": frozen_names,
        "trainable_groups": summarise(trainable_names),
        "frozen_groups": summarise(frozen_names),
        "n_trainable_params": n_trainable,
        "n_frozen_params": n_frozen,
        "trainable_fraction": n_trainable / max(1, n_trainable + n_frozen),
    }


def verify_frozen_unchanged(model: nn.Module, snapshot: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Prove after training that no frozen tensor moved."""
    max_delta = 0.0
    changed: list[str] = []
    for name, param in model.named_parameters():
        if name in snapshot:
            d = (param.detach().float().cpu() - snapshot[name]).abs().max().item()
            max_delta = max(max_delta, d)
            if d > 0.0:
                changed.append(name)
    return {
        "n_frozen_checked": len(snapshot),
        "max_abs_delta_on_frozen_params": max_delta,
        "frozen_params_that_changed": changed,
        "isolation_held": len(changed) == 0,
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Task 22 replay + parameter-isolated SFT")
    ap.add_argument("--config", default="configs/task22_replay_sft.yaml")
    ap.add_argument("--steps", type=int, default=None, help="Override max_steps (smoke tests)")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp = cfg["experiment"]
    opt_cfg = cfg["optim"]
    tr_cfg = cfg["train"]
    iso_cfg = cfg["parameter_isolation"]

    max_steps = args.steps if args.steps is not None else int(tr_cfg["max_steps"])
    seed = int(tr_cfg["seed"])
    micro_bs = int(tr_cfg["micro_batch_size"])
    accum = int(tr_cfg["grad_accum_steps"])
    eval_interval = int(tr_cfg["eval_interval"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print(" MINI-R1 -- TASK 22: OPERATOR-MATCHED REPLAY SFT + PARAMETER ISOLATION")
    print("=" * 80)
    print(f"Config      : {args.config}")
    print(f"Device      : {device}")
    print(f"Steps       : {max_steps} | micro_bs={micro_bs} accum={accum} eff_batch={micro_bs*accum} seed={seed}")
    print(f"Peak LR     : {opt_cfg['peak_lr']:.2e} (Task 21B used 1.00e-04 -- identical)")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    out_dir = exp["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best.pt")
    final_path = os.path.join(out_dir, "final.pt")

    # ---- data -------------------------------------------------------------
    tok_path = cfg["tokenizer"]["path"]
    tokenizer = Tokenizer(tok_path)
    max_seq_len = int(cfg["data"]["max_seq_len"])
    collate = lambda b: collate_sft_batch(b, pad_id=tokenizer.pad_id)  # noqa: E731
    ratio_cfg = cfg["data"].get("ratio_schedule")

    if ratio_cfg is None:
        # ---- Task 22 / 22B path: strict 1:1 two-source interleave (unchanged)
        src_a, src_b = cfg["data"]["mixture"]
        train_a = CompactReasoningDataset(src_a["train"], tokenizer, max_seq_len=max_seq_len)
        train_b = CompactReasoningDataset(src_b["train"], tokenizer, max_seq_len=max_seq_len)
        dev_a = CompactReasoningDataset(src_a["dev"], tokenizer, max_seq_len=max_seq_len)
        dev_b = CompactReasoningDataset(src_b["dev"], tokenizer, max_seq_len=max_seq_len)
        mixture = InterleavedMixtureDataset(train_a, train_b, seed=seed)
        source_specs = [src_a, src_b]
        train_sets = [train_a, train_b]
        dev_sets = [dev_a, dev_b]
        dev_weights = [0.5, 0.5]
        mixture_desc = "strict 1:1 alternation (every micro-batch is 4+4)"
        print("")
        print("Train mixture: {}={}  {}={}".format(
            src_a["name"], len(train_a), src_b["name"], len(train_b)))
        print("  interleaved epoch length = {} examples (strict 1:1 alternation)".format(len(mixture)))
    else:
        # ---- Task 23 path: deterministic global-index replay scheduling ----
        prim = cfg["data"]["primary"]
        reps = list(cfg["data"]["replay_sources"])
        period = int(ratio_cfg["period"])

        train_p = CompactReasoningDataset(prim["train"], tokenizer, max_seq_len=max_seq_len)
        dev_p = CompactReasoningDataset(prim["dev"], tokenizer, max_seq_len=max_seq_len)
        train_r = [CompactReasoningDataset(r["train"], tokenizer, max_seq_len=max_seq_len) for r in reps]
        dev_r = [CompactReasoningDataset(r["dev"], tokenizer, max_seq_len=max_seq_len) for r in reps]

        mixture = RatioScheduledMixtureDataset(train_p, train_r, period, seed)
        replay_frac = 1.0 / period
        per_src = replay_frac / len(train_r)
        source_specs = [prim] + reps
        train_sets = [train_p] + train_r
        dev_sets = [dev_p] + dev_r
        dev_weights = [1.0 - replay_frac] + [per_src] * len(train_r)
        mixture_desc = ("deterministic global-index schedule: is_replay = (idx % {} == 0); "
                        "replay slots alternate across {} sources".format(period, len(train_r)))
        print("")
        print("Task 23 ratio schedule: is_replay = (idx % {} == 0)".format(period))
        print("  primary {}: {} examples -> {:.0f}% of slots".format(
            prim["name"], len(train_p), 100 * (1 - replay_frac)))
        for r, ds in zip(reps, train_r):
            print("  replay  {}: {} examples -> {:.1f}% of slots".format(
                r["name"], len(ds), 100 * per_src))
        print("  epoch length = {} ({} primary + {} replay)".format(
            len(mixture), mixture.n_primary, mixture.n_replay))
    print("Dev sets     : " + "  ".join(
        "{}={}".format(sp["name"], len(ds)) for sp, ds in zip(source_specs, dev_sets)))

    train_loader = DataLoader(mixture, batch_size=micro_bs, shuffle=False, collate_fn=collate)
    dev_loaders = [DataLoader(ds, batch_size=micro_bs * 2, shuffle=False, collate_fn=collate)
                   for ds in dev_sets]

    # ---- model ------------------------------------------------------------
    init_ckpt_path = cfg["init"]["checkpoint"]
    init_sha = compute_file_sha256(init_ckpt_path)
    expected = cfg["init"].get("expected_sha256")
    print(f"\nInit checkpoint : {init_ckpt_path}")
    print(f"  SHA256        : {init_sha}")
    if expected:
        match = init_sha == expected
        print(f"  SHA256 matches config expectation: {match}")
        if not match:
            raise SystemExit("Init checkpoint SHA256 mismatch -- refusing to train.")

    init_ckpt = torch.load(init_ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = MiniR1Config.from_dict(init_ckpt.get("model_config", init_ckpt.get("config", {})))
    model = MiniR1Transformer(model_cfg).to(device)
    model.load_state_dict(init_ckpt.get("model_state_dict", init_ckpt.get("model", {})))
    print(f"  Loaded. Params: {model.get_num_params():,} | tie_word_embeddings={getattr(model_cfg, 'tie_word_embeddings', None)}")

    # ---- parameter isolation ---------------------------------------------
    iso = apply_parameter_isolation(model, list(iso_cfg["trainable_patterns"]))
    print("\n" + "-" * 80)
    print("PARAMETER ISOLATION")
    print("-" * 80)
    print(f"  Trainable : {iso['n_trainable_params']:,} params ({iso['trainable_fraction']*100:.2f}%)")
    for g in iso["trainable_groups"]:
        print(f"      + {g}")
    print(f"  Frozen    : {iso['n_frozen_params']:,} params ({(1-iso['trainable_fraction'])*100:.2f}%)")
    for g in iso["frozen_groups"]:
        print(f"      - {g}")
    if getattr(model_cfg, "tie_word_embeddings", False):
        print("  NOTE: weights are tied -- freezing tok_emb also freezes lm_head.")

    frozen_snapshot = {
        n: p.detach().float().cpu().clone()
        for n, p in model.named_parameters()
        if not p.requires_grad
    }

    # ---- optimizer: trainable parameters only -----------------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(opt_cfg["peak_lr"]),
        betas=tuple(opt_cfg["betas"]),
        eps=float(opt_cfg["eps"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )
    print(f"  AdamW sees {sum(p.numel() for p in trainable):,} parameters across {len(trainable)} tensors")

    use_amp = bool(tr_cfg["amp"]) and "cuda" in device
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    peak_lr = float(opt_cfg["peak_lr"])
    min_lr = float(opt_cfg["min_lr"])
    warmup = int(opt_cfg["warmup_steps"])
    max_grad_norm = float(opt_cfg["max_grad_norm"])

    def eval_all() -> dict:
        out = {}
        mix = 0.0
        for sp, dl, w in zip(source_specs, dev_loaders, dev_weights):
            l, ppl = evaluate_dev_loss(model, dl, device)
            out["{}_dev_loss".format(sp["name"])] = l
            out["{}_dev_ppl".format(sp["name"])] = ppl
            mix += w * l
        # stable aliases used by the logging/print paths
        out["gsm8k_dev_loss"] = out.get("compact_gsm8k_dev_loss", 0.0)
        out["gsm8k_dev_ppl"] = out.get("compact_gsm8k_dev_ppl", 0.0)
        rep_keys = [sp["name"] for sp in source_specs[1:]]
        if rep_keys:
            out["replay_dev_loss"] = sum(out["{}_dev_loss".format(k)] for k in rep_keys) / len(rep_keys)
            out["replay_dev_ppl"] = math.exp(min(20.0, out["replay_dev_loss"]))
        out["mixture_dev_loss"] = mix
        out["mixture_dev_ppl"] = math.exp(min(20.0, mix))
        return out

    init_ev = eval_all()
    print("\nZero-shot dev (Task 20 init, before any Task 22 update):")
    print(f"  GSM8K compact : loss {init_ev['gsm8k_dev_loss']:.4f}  ppl {init_ev['gsm8k_dev_ppl']:.2f}")
    print(f"  Replay arith  : loss {init_ev['replay_dev_loss']:.4f}  ppl {init_ev['replay_dev_ppl']:.2f}")
    print(f"  Mixture       : loss {init_ev['mixture_dev_loss']:.4f}  ppl {init_ev['mixture_dev_ppl']:.2f}")

    # ---- train ------------------------------------------------------------
    best_mix = float("inf")
    best_step = 0
    history: list[dict[str, Any]] = [{"step": 0, **init_ev, "is_best": False}]
    step = 0
    epoch = 0
    running_loss = 0.0
    since_log = 0
    total_active_tokens = 0
    start = time.time()
    last_ev = init_ev
    grad_norm = torch.tensor(0.0)
    optimizer.zero_grad(set_to_none=True)

    print("\n" + "-" * 80)
    print("TRAINING")
    print("-" * 80)

    while step < max_steps:
        mixture.reshuffle(epoch)
        epoch += 1
        model.train()
        for micro_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            total_active_tokens += batch["active_completion_tokens"]

            lr = get_cosine_lr(step, max_steps, peak_lr, min_lr, warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)) / accum

            scaler.scale(loss).backward()
            running_loss += loss.item() * accum
            since_log += 1

            if (micro_idx + 1) % accum == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % 10 == 0 or step == 1 or step == max_steps:
                    tps = total_active_tokens / max(1e-6, time.time() - start)
                    print(f"Step {step:04d}/{max_steps} | Loss: {running_loss / max(1, since_log) * accum / accum:.4f} "
                          f"| LR: {lr:.2e} | Grad: {float(grad_norm):.3f} | Tok/s: {tps:.0f}")
                    running_loss = 0.0
                    since_log = 0

                if step % eval_interval == 0 or step == max_steps:
                    ev = eval_all()
                    last_ev = ev
                    print(f"--- [EVAL {step:04d}] mix {ev['mixture_dev_loss']:.4f} "
                          f"| gsm8k {ev['gsm8k_dev_loss']:.4f} (ppl {ev['gsm8k_dev_ppl']:.2f}) "
                          f"| replay {ev['replay_dev_loss']:.4f} (ppl {ev['replay_dev_ppl']:.2f}) ---")
                    is_best = ev["mixture_dev_loss"] < best_mix
                    if is_best:
                        best_mix = ev["mixture_dev_loss"]
                        best_step = step
                        tmp = best_path + ".tmp"
                        torch.save({
                            "step": step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "model_config": model_cfg.to_dict(),
                            "dev_loss": ev["mixture_dev_loss"],
                            "dev_ppl": ev["mixture_dev_ppl"],
                            "per_source_dev": ev,
                            "init_checkpoint_sha256": init_sha,
                            "tokenizer_sha256": compute_file_sha256(tok_path),
                            "model_name": exp["model_label"],
                            "parameter_isolation": {
                                k: v for k, v in iso.items()
                                if k not in ("trainable_param_names", "frozen_param_names")
                            },
                        }, tmp)
                        os.replace(tmp, best_path)
                        print(f"==> NEW BEST -> {best_path} (mixture dev loss {best_mix:.4f})")
                    history.append({"step": step, **ev, "lr": lr, "is_best": is_best})
                    model.train()

                if step >= max_steps:
                    break

    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_cfg.to_dict(),
        "dev_loss": last_ev["mixture_dev_loss"],
        "dev_ppl": last_ev["mixture_dev_ppl"],
        "per_source_dev": last_ev,
        "init_checkpoint_sha256": init_sha,
        "tokenizer_sha256": compute_file_sha256(tok_path),
        "model_name": exp["model_label"],
    }, final_path)

    # ---- prove the isolation held ----------------------------------------
    audit = verify_frozen_unchanged(model, frozen_snapshot)
    print("\n" + "-" * 80)
    print("PARAMETER-ISOLATION AUDIT (post-training)")
    print("-" * 80)
    print(f"  frozen tensors checked          : {audit['n_frozen_checked']}")
    print(f"  max |delta| on any frozen param : {audit['max_abs_delta_on_frozen_params']:.3e}")
    print(f"  isolation held                  : {audit['isolation_held']}")
    if not audit["isolation_held"]:
        print(f"  CHANGED: {audit['frozen_params_that_changed']}")

    # base checkpoint must be untouched
    post_init_sha = compute_file_sha256(init_ckpt_path)
    print(f"  init checkpoint SHA256 unchanged: {post_init_sha == init_sha}")

    result = {
        "task": "Task 22: Operator-Matched Replay SFT with Parameter Isolation",
        "config_file": args.config,
        "model_label": exp["model_label"],
        "init_checkpoint": init_ckpt_path,
        "init_checkpoint_sha256": init_sha,
        "init_checkpoint_sha256_after_run": post_init_sha,
        "best_checkpoint": best_path,
        "best_checkpoint_sha256": compute_file_sha256(best_path),
        "final_checkpoint_sha256": compute_file_sha256(final_path),
        "best_step": best_step,
        "best_mixture_dev_loss": best_mix,
        "zero_shot_dev": init_ev,
        "final_dev": last_ev,
        "hyperparameters": {
            "max_steps": max_steps, "seed": seed,
            "micro_batch_size": micro_bs, "grad_accum_steps": accum,
            "effective_batch_size": micro_bs * accum,
            "peak_lr": peak_lr, "min_lr": min_lr, "warmup_steps": warmup,
            "weight_decay": float(opt_cfg["weight_decay"]),
            "max_grad_norm": max_grad_norm, "max_seq_len": max_seq_len,
        },
        "mixture": {
            "sources": [
                {"name": sp["name"], "train_examples": len(tr), "dev_examples": len(dv),
                 "dev_weight": w}
                for sp, tr, dv, w in zip(source_specs, train_sets, dev_sets, dev_weights)
            ],
            "interleave": mixture_desc,
            "epoch_length": len(mixture),
            "ratio_schedule": ratio_cfg,
            "replay_fraction": (1.0 / int(ratio_cfg["period"])) if ratio_cfg else 0.5,
        },
        "parameter_isolation": iso,
        "isolation_audit": audit,
        "total_active_completion_tokens": total_active_tokens,
        "history": history,
    }

    os.makedirs("reports", exist_ok=True)
    with open(exp["training_log"], "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in result.items() if k != "history"}, f, indent=2)

    print("\n" + "=" * 80)
    print("TASK 22 TRAINING COMPLETE")
    print(f"  best step            : {best_step}")
    print(f"  best mixture dev loss: {best_mix:.4f}")
    print(f"  GSM8K dev loss       : {init_ev['gsm8k_dev_loss']:.4f} -> {last_ev['gsm8k_dev_loss']:.4f}")
    print(f"  replay dev loss      : {init_ev['replay_dev_loss']:.4f} -> {last_ev['replay_dev_loss']:.4f}")
    print(f"  checkpoint           : {best_path}")
    print(f"  training log         : {exp['training_log']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
