"""
Mini-R1 -- Preprocessing Module
===============================
Provides minimal, lossless text cleaning and unified formatting for mathematical
reasoning datasets (GSM8K, SVAMP, AQuA).

Unified Format:
    Question: {question}
    <think>
    {reasoning}
    </think>
    <answer>{answer}</answer>
"""

from __future__ import annotations

import re
from typing import Any


def clean_whitespace(text: str) -> str:
    """
    Normalize line endings and collapse excessive consecutive blank lines
    while preserving math formatting, equations, and indentation.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ newlines into 2 (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_unified_text(question: str, reasoning: str, answer: str) -> str:
    """
    Construct the standard Mini-R1 reasoning format with explicit delimiters.
    """
    q_clean = clean_whitespace(question)
    r_clean = clean_whitespace(reasoning)
    a_clean = clean_whitespace(str(answer))

    return (
        f"Question: {q_clean}\n"
        f"<think>\n"
        f"{r_clean}\n"
        f"</think>\n"
        f"<answer>{a_clean}</answer>"
    )


def format_gsm8k(record: dict[str, Any], record_id: str, split: str) -> dict[str, Any]:
    """
    Convert raw GSM8K record into the unified schema.
    Raw fields: 'question', 'answer'
    """
    raw_question = record.get("question", "")
    raw_answer = record.get("answer", "")

    if "####" in raw_answer:
        reasoning_part, final_ans_part = raw_answer.split("####", 1)
    else:
        reasoning_part = raw_answer
        final_ans_part = ""

    # Clean calculator annotations (e.g. '<<48/2=24>>24' -> '24')
    reasoning_clean = re.sub(r"<<.*?>>", "", reasoning_part).strip()
    reasoning_clean = clean_whitespace(reasoning_clean)
    final_ans_clean = final_ans_part.strip()

    formatted_text = build_unified_text(raw_question, reasoning_clean, final_ans_clean)

    return {
        "dataset": "gsm8k",
        "id": record_id,
        "split": split,
        "question": clean_whitespace(raw_question),
        "reasoning": reasoning_clean,
        "answer": final_ans_clean,
        "formatted_text": formatted_text,
    }


def format_svamp(record: dict[str, Any], record_id: str, split: str) -> dict[str, Any]:
    """
    Convert raw SVAMP record into the unified schema.
    Raw fields: 'ID', 'Body', 'Question', 'Equation', 'Answer', 'Type'
    """
    body = record.get("Body", "").strip()
    q = record.get("Question", "").strip()
    full_question = f"{body} {q}".strip() if body else q

    equation = record.get("Equation", "").strip()
    answer_raw = str(record.get("Answer", "")).strip()

    # Create step-by-step reasoning explaining the arithmetic equation
    reasoning = f"To solve this problem, we evaluate the arithmetic expression: {equation} = {answer_raw}."

    formatted_text = build_unified_text(full_question, reasoning, answer_raw)

    return {
        "dataset": "svamp",
        "id": record_id or str(record.get("ID", "")),
        "split": split,
        "question": clean_whitespace(full_question),
        "reasoning": clean_whitespace(reasoning),
        "answer": answer_raw,
        "formatted_text": formatted_text,
    }


def format_aqua(record: dict[str, Any], record_id: str, split: str) -> dict[str, Any]:
    """
    Convert raw AQuA-RAT record into the unified schema.
    Raw fields: 'question', 'options' (list of 'A)...', 'B)...'), 'rationale', 'correct' ('A'..'E')
    """
    question = record.get("question", "").strip()
    options = record.get("options", [])
    if isinstance(options, (list, tuple)) and options:
        opt_str = "\nOptions:\n" + "\n".join(f"- {opt}" for opt in options)
        full_question = f"{question}\n{opt_str}".strip()
    else:
        full_question = question

    rationale = clean_whitespace(record.get("rationale", ""))
    correct = str(record.get("correct", "")).strip()

    formatted_text = build_unified_text(full_question, rationale, correct)

    return {
        "dataset": "aqua",
        "id": record_id,
        "split": split,
        "question": clean_whitespace(full_question),
        "reasoning": rationale,
        "answer": correct,
        "formatted_text": formatted_text,
    }
