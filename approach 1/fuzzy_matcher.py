"""
Fuzzy query matching for handling typos, paraphrasing, and grammatical variations.
Uses multiple strategies to match student queries to FAQ entries.
"""

from typing import List, Tuple
from rapidfuzz import fuzz, process
from processor import normalize_text


_KEYWORD_SYNONYMS = {
    "fee": {"fees", "cost", "charges", "expense", "tuition", "payment"},
    "structure": {"breakdown", "details", "amount", "rates", "schedule", "structure"},
    "deadline": {"due date", "last date", "submission date", "closing date"},
    "eligibility": {"requirement", "qualification", "criteria", "condition"},
    "hostel": {"accommodation", "residence", "dormitory", "housing"},
    "admission": {"admissions", "apply", "application", "enroll"},
    "programme": {"program", "course", "degree", "major"},
    "nust": {"nust", "nuist", "nust pakistan"},  # Common typos
}


def expand_keywords(text: str) -> str:
    """Expand query with synonym keywords to improve matching."""
    tokens = text.lower().split()
    expanded = []
    
    for token in tokens:
        expanded.append(token)
        # Check if this token matches any synonym group
        for main_keyword, synonyms in _KEYWORD_SYNONYMS.items():
            if token == main_keyword or token in synonyms:
                expanded.extend(list(synonyms)[:2])  # Add top synonyms
                break
    
    return " ".join(expanded)


def fuzzy_match_query_to_faq(
    user_query: str,
    faq_questions: List[str],
    threshold: int = 70
) -> Tuple[str, int] | None:
    """
    Fuzzy match user query against FAQ questions.
    Returns (best_faq_question, match_score) or None if no good match.
    
    This handles:
    - Typos (NSUT → NUST)
    - Paraphrasing (fee vs cost)
    - Word order (structure fee vs fee structure)
    - Grammatical variations (What is fee vs fees)
    """
    if not user_query or not faq_questions:
        return None
    
    normalized_query = normalize_text(user_query)
    expanded_query = expand_keywords(normalized_query)
    
    # Try different match strategies
    
    # Strategy 1: Token set ratio (handles word order, extra words)
    best_match = process.extractOne(
        expanded_query,
        faq_questions,
        scorer=fuzz.token_set_ratio,
        score_cutoff=max(threshold - 10, 60)  # More lenient for paraphrasing
    )
    
    if best_match:
        return (best_match[0], best_match[1])
    
    # Strategy 2: Partial ratio (handles substring matches)
    best_match = process.extractOne(
        normalized_query,
        faq_questions,
        scorer=fuzz.partial_ratio,
        score_cutoff=max(threshold - 15, 55)  # More lenient
    )
    
    if best_match:
        return (best_match[0], best_match[1])
    
    # Strategy 3: Simple ratio as fallback
    best_match = process.extractOne(
        normalized_query,
        faq_questions,
        scorer=fuzz.ratio,
        score_cutoff=max(threshold - 8, 70)
    )
    
    if best_match:
        return (best_match[0], best_match[1])
    
    # Strategy 4: Keyword overlap fallback for short queries (conservative).
    query_keywords = {tok for tok in normalized_query.split() if len(tok) > 2}
    if 1 <= len(query_keywords) <= 4:
        best_score = 0
        best_question = None
        for faq_q in faq_questions:
            faq_keywords = set(faq_q.split())
            overlap = len(query_keywords.intersection(faq_keywords))
            if overlap >= 2:
                score = int((overlap / max(len(query_keywords), 1)) * 100)
                if score > best_score:
                    best_score = score
                    best_question = faq_q

        if best_question and best_score >= max(threshold, 80):
            return (best_question, best_score)
    
    return None


def apply_typo_correction(text: str) -> str:
    """
    Correct common typos in admissions queries.
    """
    corrections = {
        "nsut": "nust",           # NSUT → NUST
        "nuist": "nust",          # NUIST → NUST
        "whaat": "what",
        "wht": "what",
        "sturcture": "structure",
        "strcture": "structure",
        "stucture": "structure",
        "bsnhd": "bshnsd",        # Common NUST program abbreviations
        "fee structer": "fee structure",
        "programm": "programme",
        "programe": "programme",
    }
    
    text_lower = text.lower()
    for typo, correct in corrections.items():
        text_lower = text_lower.replace(typo, correct)
    
    return text_lower

