from dataclasses import dataclass
from typing import List

from retriever import RetrievalHit


NOT_FOUND_MESSAGE = "I'm not able to find this information in the official NUST FAQs."


@dataclass(frozen=True)
class QueryResult:
    sub_query: str
    found: bool
    hit: RetrievalHit | None


def build_response(results: List[QueryResult]) -> str:
    if not results:
        return "Please ask a question so I can look it up in the official FAQs."

    found_results = [result for result in results if result.found and result.hit is not None]
    missing_results = [result for result in results if not result.found]

    if not found_results:
        return NOT_FOUND_MESSAGE

    if len(results) == 1 and len(found_results) == 1 and not missing_results:
        return f"The information I found is:\n\n{found_results[0].hit.entry.answer}"

    bullets = [
        f"- {result.hit.entry.question}: {result.hit.entry.answer}"
        for result in found_results
        if result.hit is not None
    ]

    if missing_results and found_results:
        missing_topics = ", ".join(result.sub_query for result in missing_results)
        return (
            f"I could not find information about {missing_topics}, but here is what I found:\n\n"
            + "\n".join(bullets)
        )

    return "Here's what I found:\n\n" + "\n".join(bullets)

