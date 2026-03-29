from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import List

from data_loader import FAQEntry, load_faq_entries
from processor import process_query
from responder import QueryResult, build_response
from retriever import HybridRetriever


@dataclass(frozen=True)
class ChatbotConfig:
    faq_path: Path
    score_threshold: float = 0.62
    min_keyword_score: float = 0.05
    min_semantic_score: float = 0.82
    top_k: int = 5


class OfflineAdmissionsChatbot:
    def __init__(self, config: ChatbotConfig) -> None:
        self.config = config
        self.entries: List[FAQEntry] = load_faq_entries(str(config.faq_path))
        self.retriever = HybridRetriever(self.entries, local_files_only=True)

    def answer(self, query: str) -> str:
        sub_queries = process_query(query)
        if not sub_queries:
            return "Please enter a question."

        results: List[QueryResult] = []
        for sub_query in sub_queries:
            hits = self.retriever.retrieve(sub_query, top_k=self.config.top_k)
            top_hit = hits[0] if hits else None
            is_confident = bool(
                top_hit
                and top_hit.final_score >= self.config.score_threshold
                and (
                    top_hit.keyword_score >= self.config.min_keyword_score
                    or top_hit.semantic_score >= self.config.min_semantic_score
                )
            )
            results.append(QueryResult(sub_query=sub_query, found=is_confident, hit=top_hit))

        return build_response(results)


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

