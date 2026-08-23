"""
test_arithmetic_dataset.py - Comprehensive Unit Tests for Task 18A Verified Arithmetic Dataset.
"""

import json
import os
import pytest

from scripts.verify_arithmetic_dataset import extract_answer_tag, verify_single_record


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "arithmetic_sft")


def test_files_exist_and_manifest():
    """Verify all 4 jsonl files and manifest.json exist."""
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    assert os.path.exists(manifest_path), "manifest.json missing"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    assert manifest["total_train_examples"] == 12000
    assert manifest["total_dev_examples"] == 1200
    assert "curriculum_levels" in manifest
    assert "operand_ranges" in manifest


@pytest.mark.parametrize("filename, expected_rows", [
    ("compact_train.jsonl", 12000),
    ("compact_dev.jsonl", 1200),
    ("verbose_train.jsonl", 12000),
    ("verbose_dev.jsonl", 1200),
])
def test_dataset_integrity_and_math(filename, expected_rows):
    """Verify row count, tags, and 100% math validity across all rows."""
    file_path = os.path.join(DATA_DIR, filename)
    assert os.path.exists(file_path), f"{filename} does not exist"

    row_count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row_count += 1
            record = json.loads(line)
            is_valid, err_msg = verify_single_record(record)
            assert is_valid, f"Error in {filename} row {row_count}: {err_msg}"

    assert row_count == expected_rows, f"Expected {expected_rows} rows, got {row_count}"


def test_paired_compact_and_verbose_consistency():
    """Verify compact and verbose train sets share identical problems and answers."""
    c_path = os.path.join(DATA_DIR, "compact_train.jsonl")
    v_path = os.path.join(DATA_DIR, "verbose_train.jsonl")

    with open(c_path, "r", encoding="utf-8") as fc, open(v_path, "r", encoding="utf-8") as fv:
        for c_line, v_line in zip(fc, fv):
            c_rec = json.loads(c_line)
            v_rec = json.loads(v_line)

            assert c_rec["example_id"] == v_rec["example_id"]
            assert c_rec["prompt"] == v_rec["prompt"]
            
            c_ans = extract_answer_tag(c_rec["completion"])
            v_ans = extract_answer_tag(v_rec["completion"])
            assert c_ans == v_ans, f"Mismatch in answers for {c_rec['example_id']}: {c_ans} vs {v_ans}"
            assert c_ans == c_rec["metadata"]["final_answer"]
