import re
from typing import List

_COMMA_ALSO_SPLIT_REGEX = re.compile(r"\s*(?:,|\balso\b)\s*", re.IGNORECASE)
_PUNCT_REGEX = re.compile(r"[^a-z0-9\s]")
_SPACE_REGEX = re.compile(r"\s+")
_QUESTION_CUES = {
    "what",
    "how",
    "is",
    "are",
    "can",
    "do",
    "does",
    "where",
    "when",
    "which",
    "who",
    "whom",
    "will",
}


def normalize_text(text: str) -> str:
    """Lowercase and remove punctuation for stable matching."""
    lowered = text.lower().strip()
    no_punct = _PUNCT_REGEX.sub(" ", lowered)
    return _SPACE_REGEX.sub(" ", no_punct).strip()


def tokenize(text: str) -> List[str]:
    """Tokenize normalized text for BM25 indexing."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split(" ")


def split_compound_query(query: str, max_parts: int = 5) -> List[str]:
    """Split user query into sub-queries using light rule-based separators."""
    lowered_query = (query or "").lower().strip()
    if not lowered_query:
        return []

    # Keep commas for boundary detection, then normalize each split fragment.
    split_ready_query = _SPACE_REGEX.sub(" ", lowered_query)

    coarse_parts = _COMMA_ALSO_SPLIT_REGEX.split(split_ready_query)
    parts: List[str] = []

    for coarse in coarse_parts:
        text = normalize_text(coarse)
        if not text:
            continue
        parts.extend(_split_on_and_if_compound(text))

    # Keep order and remove exact duplicates for deterministic behavior.
    deduped_parts: List[str] = []
    seen = set()
    for part in parts:
        if part not in seen:
            seen.add(part)
            deduped_parts.append(part)
        if len(deduped_parts) >= max_parts:
            break

    normalized_query = normalize_text(query)
    return deduped_parts if deduped_parts else [normalized_query]


def _split_on_and_if_compound(text: str) -> List[str]:
    if " and " not in text:
        return [text]

    left, right = text.split(" and ", 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return [text]

    left_tokens = left.split()
    right_tokens = right.split()

    # Only split when both sides look like independent question fragments.
    if (
        len(left_tokens) >= 3
        and len(right_tokens) >= 3
        and any(token in _QUESTION_CUES for token in left_tokens)
        and any(token in _QUESTION_CUES for token in right_tokens)
    ):
        return [left, right]

    return [text]


def process_query(query: str, max_chars: int = 500) -> List[str]:
    """Validate and split user query before retrieval."""
    trimmed = (query or "").strip()
    if not trimmed:
        return []

    # Avoid pathological latency on extremely long prompts.
    safe_query = trimmed[:max_chars]
    return split_compound_query(safe_query)

