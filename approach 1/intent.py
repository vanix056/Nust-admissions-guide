
from processor import normalize_text


_INTENT_RESPONSES = {
    "greeting": "Hello! I'm here to help you with NUST admissions. How can I assist you today?",
    "how_are_you": "I'm functioning well and ready to assist you with NUST admissions information.",
    "bot_identity": "I'm an offline NUST admissions assistant. I provide information based on official FAQ data.",
    "help": "I can help you with questions about NUST admissions, such as deadlines, eligibility, and fee structure.",
    "thanks": "You're welcome. Let me know if you need anything else.",
    "goodbye": "Goodbye. Feel free to return if you have more questions about admissions.",
}

_GREETING_PATTERNS = ("hi", "hello", "hey", "assalamualaikum")
_HOW_ARE_YOU_PATTERNS = ("how are you", "how are you doing")
_BOT_IDENTITY_PATTERNS = ("who are you", "what are you", "your job", "what do you do")
_HELP_PATTERNS = ("help", "what can you do", "how can you help", "can you help me")
_THANKS_PATTERNS = ("thanks", "thank you")
_GOODBYE_PATTERNS = ("bye", "goodbye")

_FAQ_HINT_KEYWORDS = {
    "admission",
    "admissions",
    "deadline",
    "eligibility",
    "fee",
    "fees",
    "hostel",
    "test",
    "net",
    "programme",
    "program",
    "apply",
    "merit",
    "undergraduate",
    "postgraduate",
    "mbbs",
}
_QUESTION_WORDS = {"what", "when", "where", "which", "who", "how", "can", "is", "are", "do", "does"}


def _contains_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _has_faq_keywords(normalized_query: str) -> bool:
    return bool(set(normalized_query.split()).intersection(_FAQ_HINT_KEYWORDS))


def _looks_like_faq_query(raw_query: str, normalized_query: str) -> bool:
    tokens = normalized_query.split()

    if len(raw_query.strip()) > 35:
        return True
    if len(tokens) > 6:
        return True
    if _has_faq_keywords(normalized_query):
        return True

    token_set = set(tokens)
    if "?" in raw_query and token_set.intersection(_QUESTION_WORDS):
        return True

    return False


def detect_intent(query: str) -> str:
    normalized = normalize_text(query)
    if not normalized:
        return "faq_query"

    is_faq_like = _looks_like_faq_query(query or "", normalized)
    has_faq_keywords = _has_faq_keywords(normalized)

    if _contains_any_phrase(normalized, _HOW_ARE_YOU_PATTERNS):
        return "faq_query" if has_faq_keywords else "how_are_you"

    if _contains_any_phrase(normalized, _BOT_IDENTITY_PATTERNS):
        return "faq_query" if has_faq_keywords else "bot_identity"

    if _contains_any_phrase(normalized, _HELP_PATTERNS):
        return "faq_query" if has_faq_keywords else "help"

    if _contains_any_phrase(normalized, _THANKS_PATTERNS):
        return "faq_query" if is_faq_like else "thanks"

    if _contains_any_phrase(normalized, _GOODBYE_PATTERNS):
        return "faq_query" if is_faq_like else "goodbye"

    if _contains_any_phrase(normalized, _GREETING_PATTERNS):
        return "faq_query" if is_faq_like else "greeting"

    return "faq_query"


def get_conversational_response(intent: str) -> str:
    return _INTENT_RESPONSES.get(intent, "")


