import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from processor import normalize_text, tokenize


@dataclass(frozen=True)
class FAQEntry:
    question: str
    answer: str
    normalized_question: str
    question_tokens: List[str]


def load_faq_entries(json_path: str) -> List[FAQEntry]:
    """Load FAQ records and keep only valid entries."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"FAQ file not found: {json_path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("FAQ JSON must be a list of objects.")

    entries: List[FAQEntry] = []
    seen = set()

    for row in data:
        if not isinstance(row, dict):
            continue
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            continue

        normalized_q = normalize_text(question)
        # Remove duplicate question+answer pairs to reduce index noise.
        dedupe_key = (normalized_q, answer)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        entries.append(
            FAQEntry(
                question=question,
                answer=answer,
                normalized_question=normalized_q,
                question_tokens=tokenize(question),
            )
        )

    if not entries:
        raise ValueError("No valid FAQ records found in JSON file.")

    return entries

