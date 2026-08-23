"""
Mini-R1 -- Dataset Pipeline Test Suite (Task 3)
==============================================
Tests verifying:
1. Dataset loading across GSM8K, SVAMP, and AQuA.
2. Unified formatting (<think> and <answer> tags present and non-empty).
3. Split separation (train, validation, and test splits are mutually disjoint).
4. Strict zero test-set leakage.
5. Preprocessing unit tests.
"""

from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.datasets import ReasoningDatasetManager, DATASET_SOURCES
from src.preprocessing import clean_whitespace, format_gsm8k, format_svamp, format_aqua, build_unified_text


@pytest.fixture(scope="session")
def dataset_manager() -> ReasoningDatasetManager:
    """Session-level dataset manager fixture."""
    return ReasoningDatasetManager(seed=42, val_ratio=0.1)


class TestPreprocessingUnit:
    """Unit tests for string cleaning and formatters."""

    def test_clean_whitespace(self):
        """Line endings and excessive blank lines are cleaned without destroying structure."""
        raw = "Line 1\r\n\r\n\r\n\r\nLine 2\r\n"
        cleaned = clean_whitespace(raw)
        assert cleaned == "Line 1\n\nLine 2"

    def test_build_unified_text(self):
        """Unified text format contains required headers and XML reasoning tags."""
        text = build_unified_text("What is 1+1?", "1+1=2", "2")
        assert text.startswith("Question: What is 1+1?\n<think>\n1+1=2\n</think>\n<answer>2</answer>")

    def test_format_gsm8k(self):
        """GSM8K record formatting cleans calculator annotations and splits answer."""
        raw = {
            "question": "Janet has 3 cats.",
            "answer": "Janet has 3*2 = <<3*2=6>>6 legs.\n#### 6",
        }
        res = format_gsm8k(raw, record_id="gsm8k_0", split="train")
        assert res["dataset"] == "gsm8k"
        assert res["answer"] == "6"
        assert "<<" not in res["reasoning"]
        assert "<think>" in res["formatted_text"]
        assert "<answer>6</answer>" in res["formatted_text"]

    def test_format_svamp(self):
        """SVAMP record concatenates body and question and formats equation."""
        raw = {
            "ID": "svamp-1",
            "Body": "Tom has 10 apples.",
            "Question": "He eats 3. How many left?",
            "Equation": "( 10.0 - 3.0 )",
            "Answer": "7",
        }
        res = format_svamp(raw, record_id="svamp_0", split="train")
        assert res["dataset"] == "svamp"
        assert "Tom has 10 apples. He eats 3." in res["question"]
        assert res["answer"] == "7"
        assert "<think>" in res["formatted_text"]
        assert "<answer>7</answer>" in res["formatted_text"]

    def test_format_aqua(self):
        """AQuA record includes multiple-choice options and rationale."""
        raw = {
            "question": "What is 2+2?",
            "options": ["A)1", "B)2", "C)4", "D)5"],
            "rationale": "2+2=4. So option C.",
            "correct": "C",
        }
        res = format_aqua(raw, record_id="aqua_0", split="train")
        assert res["dataset"] == "aqua"
        assert "Options:" in res["question"]
        assert res["answer"] == "C"
        assert "<think>" in res["formatted_text"]
        assert "<answer>C</answer>" in res["formatted_text"]


class TestDatasetManagerLoading:
    """Verify loading and split counts across datasets."""

    @pytest.mark.parametrize("dataset_name", ["gsm8k", "svamp", "aqua"])
    def test_dataset_loads_all_splits(self, dataset_manager: ReasoningDatasetManager, dataset_name: str):
        """Each dataset must load non-empty train, validation, and test splits."""
        train_recs = dataset_manager.get_formatted_split(dataset_name, split="train", limit=5)
        val_recs = dataset_manager.get_formatted_split(dataset_name, split="validation", limit=5)
        test_recs = dataset_manager.get_formatted_split(dataset_name, split="test", limit=5)

        assert len(train_recs) == 5
        assert len(val_recs) == 5
        assert len(test_recs) == 5

        # Check required keys in formatted dictionary
        for r in train_recs:
            assert "question" in r and len(r["question"]) > 0
            assert "reasoning" in r and len(r["reasoning"]) > 0
            assert "answer" in r and len(r["answer"]) > 0
            assert "formatted_text" in r and "<think>" in r["formatted_text"]


class TestSplitSeparationAndNoLeakage:
    """Verify train, validation, and test splits are strictly disjoint (zero test leakage)."""

    @pytest.mark.parametrize("dataset_name", ["gsm8k", "svamp", "aqua"])
    def test_no_overlap_between_splits(self, dataset_manager: ReasoningDatasetManager, dataset_name: str):
        """Train, validation, and test question sets must have zero overlap."""
        train_raw = dataset_manager.get_split_raw(dataset_name, "train")
        val_raw = dataset_manager.get_split_raw(dataset_name, "validation")
        test_raw = dataset_manager.get_split_raw(dataset_name, "test")

        # Extract question strings
        def get_q(item: dict) -> str:
            if "question" in item:
                return item["question"].strip()
            if "Question" in item:
                return f"{item.get('Body', '')} {item['Question']}".strip()
            return str(item)

        train_qs = {get_q(x) for x in train_raw}
        val_qs = {get_q(x) for x in val_raw}
        test_qs = {get_q(x) for x in test_raw}

        # Check train vs validation
        train_val_overlap = train_qs.intersection(val_qs)
        assert len(train_val_overlap) == 0, f"{dataset_name}: Train and Val overlap ({len(train_val_overlap)} questions)"

        # Check train vs test (CRITICAL)
        train_test_overlap = train_qs.intersection(test_qs)
        assert len(train_test_overlap) == 0, f"{dataset_name}: DATA LEAKAGE between Train and Test ({len(train_test_overlap)} questions)"

        # Check val vs test (CRITICAL)
        val_test_overlap = val_qs.intersection(test_qs)
        assert len(val_test_overlap) == 0, f"{dataset_name}: DATA LEAKAGE between Val and Test ({len(val_test_overlap)} questions)"


class TestCombinedBatch:
    """Verify combined batch generation."""

    def test_get_combined_train(self, dataset_manager: ReasoningDatasetManager):
        """Combined batch returns requested number of examples across datasets."""
        combined = dataset_manager.get_combined_train(
            datasets=("gsm8k", "svamp", "aqua"),
            limit_per_dataset=10,
            shuffle=True,
        )
        assert len(combined) == 30
        datasets_present = {r["dataset"] for r in combined}
        assert datasets_present == {"gsm8k", "svamp", "aqua"}
