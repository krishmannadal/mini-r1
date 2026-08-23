"""
Mini-R1 -- Dataset Management Module
====================================
Handles downloading, caching, split management, and formatted sample retrieval
for online mathematical reasoning datasets:
1. GSM8K (openai/gsm8k)
2. SVAMP (ChilleD/SVAMP)
3. AQuA-RAT (deepmind/aqua_rat)

Features:
- Unified schema integration via src.preprocessing.
- Deterministic train/val split creation for datasets lacking explicit validation splits.
- Upstream deduplication and strict filtering to guarantee zero overlap between train, val, and test.
"""

from __future__ import annotations

import random
from typing import Any, Sequence
from datasets import load_dataset, DatasetDict

from src.preprocessing import format_gsm8k, format_svamp, format_aqua


DATASET_SOURCES = {
    "gsm8k": {"path": "openai/gsm8k", "config": "main"},
    "svamp": {"path": "ChilleD/SVAMP", "config": None},
    "aqua": {"path": "deepmind/aqua_rat", "config": None},
}


def _extract_question(item: dict[str, Any]) -> str:
    """Extract normalized question text from raw dataset item."""
    if "question" in item:
        return str(item["question"]).strip()
    if "Question" in item:
        body = str(item.get("Body", "")).strip()
        q = str(item["Question"]).strip()
        return f"{body} {q}".strip() if body else q
    return ""


class ReasoningDatasetManager:
    """
    Manager for loading, splitting, and formatting mathematical reasoning datasets.
    """

    def __init__(self, seed: int = 42, val_ratio: float = 0.1) -> None:
        """
        Initialize dataset manager.

        Args:
            seed: Random seed for deterministic train/val splitting.
            val_ratio: Proportion of train split to allocate for validation
                       when a dataset does not provide a native validation split.
        """
        self.seed = seed
        self.val_ratio = val_ratio
        self._raw_cache: dict[str, DatasetDict] = {}
        self._processed_splits: dict[str, dict[str, list[dict[str, Any]]]] = {}

    def load_raw(self, name: str) -> DatasetDict:
        """
        Download and cache the raw dataset from Hugging Face.
        """
        key = name.lower()
        if key not in DATASET_SOURCES:
            raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_SOURCES.keys())}")

        if key not in self._raw_cache:
            spec = DATASET_SOURCES[key]
            if spec["config"]:
                ds = load_dataset(spec["path"], spec["config"])
            else:
                ds = load_dataset(spec["path"])
            self._raw_cache[key] = ds

        return self._raw_cache[key]

    def _prepare_splits(self, dataset_name: str) -> dict[str, list[dict[str, Any]]]:
        """
        Prepare and partition raw dataset into disjoint train, validation, and test lists of dicts.
        Filters out any upstream duplicate questions across splits to enforce 100% zero leakage.
        """
        key = dataset_name.lower()
        if key in self._processed_splits:
            return self._processed_splits[key]

        raw_ds = self.load_raw(key)

        # 1. Test Split (Golden benchmark - strictly isolated)
        raw_test = [dict(x) for x in raw_ds["test"]]
        test_questions = {_extract_question(x) for x in raw_test}

        # 2. Validation Split
        if "validation" in raw_ds:
            raw_val_all = [dict(x) for x in raw_ds["validation"]]
            # Filter out any accidental overlap with test
            raw_val = [x for x in raw_val_all if _extract_question(x) not in test_questions]
        else:
            # Deterministic split from train
            train_pool = [dict(x) for x in raw_ds["train"]]
            # Exclude any that appear in test
            train_pool = [x for x in train_pool if _extract_question(x) not in test_questions]
            rng = random.Random(self.seed)
            rng.shuffle(train_pool)
            n_val = max(1, int(len(train_pool) * self.val_ratio))
            raw_val = train_pool[:n_val]

        val_questions = {_extract_question(x) for x in raw_val}

        # 3. Train Split (Strictly filtered against Test and Validation)
        if "validation" in raw_ds:
            raw_train_all = [dict(x) for x in raw_ds["train"]]
            # Filter out any questions matching test or val
            raw_train = [
                x for x in raw_train_all
                if _extract_question(x) not in test_questions and _extract_question(x) not in val_questions
            ]
        else:
            raw_train = train_pool[n_val:]

        self._processed_splits[key] = {
            "train": raw_train,
            "validation": raw_val,
            "test": raw_test,
        }
        return self._processed_splits[key]

    def get_split_raw(self, dataset_name: str, split: str) -> list[dict[str, Any]]:
        """
        Retrieve raw examples for a specific split as a list of dicts.
        """
        splits = self._prepare_splits(dataset_name)
        if split not in splits:
            raise ValueError(f"Unknown split '{split}'. Expected 'train', 'validation', or 'test'.")
        return splits[split]

    def get_formatted_split(
        self,
        dataset_name: str,
        split: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve and format examples from a dataset split into the unified schema.
        """
        raw_examples = self.get_split_raw(dataset_name, split)
        formatter = {
            "gsm8k": format_gsm8k,
            "svamp": format_svamp,
            "aqua": format_aqua,
        }[dataset_name.lower()]

        if limit is not None:
            raw_examples = raw_examples[:limit]

        formatted_records = []
        for idx, item in enumerate(raw_examples):
            rec_id = f"{dataset_name.lower()}_{split}_{idx}"
            formatted = formatter(item, record_id=rec_id, split=split)
            formatted_records.append(formatted)

        return formatted_records

    def get_combined_train(
        self,
        datasets: Sequence[str] = ("gsm8k", "svamp", "aqua"),
        limit_per_dataset: int | None = None,
        shuffle: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Retrieve unified training records across all specified datasets.
        """
        combined = []
        for d in datasets:
            records = self.get_formatted_split(d, split="train", limit=limit_per_dataset)
            combined.extend(records)

        if shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(combined)

        return combined

    def get_statistics(self) -> dict[str, dict[str, int]]:
        """
        Calculate total and split counts across all registered datasets.
        """
        stats = {}
        for d in DATASET_SOURCES:
            splits = self._prepare_splits(d)
            stats[d] = {
                "train": len(splits["train"]),
                "validation": len(splits["validation"]),
                "test": len(splits["test"]),
            }
            stats[d]["total"] = sum(stats[d].values())
        return stats
