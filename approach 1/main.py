from dataclasses import dataclass
from pathlib import Path
import re
from time import perf_counter
from typing import List

from data_loader import FAQEntry, load_faq_entries
from intent import detect_intent, get_conversational_response
from processor import normalize_text, process_query
from responder import QueryResult, build_response
from retriever import HybridRetriever


_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "am",
    "do",
    "does",
    "for",
    "to",
    "of",
    "in",
    "on",
    "at",
    "about",
    "and",
    "or",
    "i",
    "me",
    "my",
    "we",
    "you",
    "your",
    "can",
    "could",
    "please",
    "what",
    "which",
    "when",
    "where",
    "how",
}
_EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_REGEX = re.compile(r"\+?\d[\d\-\s]{7,}\d")
_URL_REGEX = re.compile(r"https?://[^\s]+")


@dataclass(frozen=True)
class ChatbotConfig:
    faq_path: Path
    score_threshold: float = 0.7
    min_keyword_score: float = 0.1
    min_semantic_score: float = 0.9
    min_score_gap: float = 0.06
    top_k: int = 5


class OfflineAdmissionsChatbot:
    def __init__(self, config: ChatbotConfig) -> None:
        self.config = config
        self.entries: List[FAQEntry] = load_faq_entries(str(config.faq_path))
        self.retriever = HybridRetriever(self.entries, local_files_only=True)
        self.support_contact_line = self._build_support_contact_line(self.entries)

    def _build_support_contact_line(self, entries: List[FAQEntry]) -> str:
        # Prioritize records that explicitly discuss admissions contact details.
        explicit_candidates = [
            entry
            for entry in entries
            if "whom should i contact" in entry.normalized_question
            or "contact for queries" in entry.normalized_question
        ]
        contact_candidates = [
            entry
            for entry in entries
            if any(
                keyword in entry.normalized_question
                for keyword in ("contact", "queries", "admission", "entry test")
            )
        ]
        if explicit_candidates:
            scan_pool = explicit_candidates
        elif contact_candidates:
            scan_pool = contact_candidates
        else:
            scan_pool = entries

        emails: List[str] = []
        phones: List[str] = []
        urls: List[str] = []

        for entry in scan_pool:
            text = f"{entry.question} {entry.answer}".replace("[", "").replace("]", "")
            compact_text = re.sub(r"\s+", "", text).replace("\xa0", "")

            for email in _EMAIL_REGEX.findall(text):
                if email not in emails:
                    emails.append(email)
            for email in _EMAIL_REGEX.findall(compact_text):
                if email not in emails:
                    emails.append(email)

            for phone in _PHONE_REGEX.findall(text):
                clean_phone = re.sub(r"\s+", " ", phone).strip()
                if clean_phone not in phones:
                    phones.append(clean_phone)

            for url in _URL_REGEX.findall(text):
                if "nust" in url.lower() and url not in urls:
                    urls.append(url)

            if emails and phones:
                break

        parts: List[str] = []
        if phones:
            parts.append(f"Phone: {phones[0]}")
        if emails:
            parts.append(f"Email: {emails[0]}")
        if urls:
            parts.append(f"Website: {urls[0]}")

        if parts:
            return " | ".join(parts)

        return "the NUST Admissions Office through official NUST admissions channels"

    @staticmethod
    def _overlap_ok(query_text: str, matched_question: str) -> bool:
        query_tokens = set(normalize_text(query_text).split())
        question_tokens = set(normalize_text(matched_question).split())
        if not query_tokens or not question_tokens:
            return False

        query_keywords = {tok for tok in query_tokens if tok not in _STOPWORDS and len(tok) > 2}
        question_keywords = {tok for tok in question_tokens if tok not in _STOPWORDS and len(tok) > 2}

        if not query_keywords:
            query_keywords = query_tokens
        if not question_keywords:
            question_keywords = question_tokens

        overlap = len(query_keywords.intersection(question_keywords))
        required = 1 if len(query_keywords) <= 2 else max(2, int(round(0.4 * len(query_keywords))))
        return overlap >= required

    def _is_confident_hit(self, sub_query: str, hits: List) -> bool:
        top_hit = hits[0] if hits else None
        if top_hit is None:
            return False

        overlap_ok = self._overlap_ok(sub_query, top_hit.entry.question)
        score_ok = (
            top_hit.final_score >= self.config.score_threshold
            and (
                top_hit.keyword_score >= self.config.min_keyword_score
                or top_hit.semantic_score >= self.config.min_semantic_score
            )
        )

        # Require a margin from second best to avoid confident but ambiguous matches.
        margin_ok = True
        if len(hits) > 1:
            margin_ok = (top_hit.final_score - hits[1].final_score) >= self.config.min_score_gap

        return bool(overlap_ok and score_ok and margin_ok)

    def answer(self, query: str) -> str:
        intent = detect_intent(query)
        if intent != "faq_query":
            return get_conversational_response(intent)

        sub_queries = process_query(query)
        if not sub_queries:
            return "Please enter a question."

        # For mixed prompts (e.g., "hi, what is deadline"), keep only FAQ sub-queries.
        faq_sub_queries = [part for part in sub_queries if detect_intent(part) == "faq_query"]
        search_parts = faq_sub_queries if faq_sub_queries else sub_queries

        results: List[QueryResult] = []
        for sub_query in search_parts:
            hits = self.retriever.retrieve(sub_query, top_k=self.config.top_k)
            top_hit = hits[0] if hits else None
            is_confident = self._is_confident_hit(sub_query, hits)
            results.append(QueryResult(sub_query=sub_query, found=is_confident, hit=top_hit))

        return build_response(results, support_contact_line=self.support_contact_line)


def build_default_chatbot() -> OfflineAdmissionsChatbot:
    root = Path(__file__).resolve().parent
    config = ChatbotConfig(faq_path=root / "data" / "nust_faq.json")
    try:
        return OfflineAdmissionsChatbot(config)
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize offline chatbot. Make sure model 'all-MiniLM-L6-v2' is available locally."
        ) from exc


def run_cli() -> None:
    bot = build_default_chatbot()
    print("Offline NUST admissions chatbot is ready. Type 'exit' to quit.")

    while True:
        user_input = input("Ask: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        started = perf_counter()
        response = bot.answer(user_input)
        elapsed_ms = (perf_counter() - started) * 1000

        print(response)
        print(f"\n[response_time_ms={elapsed_ms:.2f}]\n")


if __name__ == "__main__":
    run_cli()

