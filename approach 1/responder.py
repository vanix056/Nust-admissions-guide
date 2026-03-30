from dataclasses import dataclass
import re
from typing import List

from retriever import RetrievalHit


NOT_FOUND_MESSAGE = "I'm not able to find this information in the official NUST FAQs."

_MULTI_SPACE_REGEX = re.compile(r"[ \t]+")
_LINE_BREAK_HINT_REGEX = re.compile(
    r"(?i)\s*(phone:|e-?mail:|email:|website:|for details|visit)"
)
_SENTENCE_BREAK_REGEX = re.compile(r"\.\s+(?=[A-Z])")
_GLUED_URL_REGEX = re.compile(r"(?i)([a-z])((?:https?://|www\.))")


def format_answer_text(answer: str) -> str:
    """Lightly format FAQ answer text for readability without changing facts."""
    text = (answer or "").strip()
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE_REGEX.sub(" ", text)
    text = _GLUED_URL_REGEX.sub(r"\1 \2", text)
    text = text.replace(";", ";\n")
    text = _LINE_BREAK_HINT_REGEX.sub(lambda match: f"\n{match.group(1)}", text)
    text = _SENTENCE_BREAK_REGEX.sub(".\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass(frozen=True)
class QueryResult:
    sub_query: str
    found: bool
    hit: RetrievalHit | None


def build_response(results: List[QueryResult], support_contact_line: str = "") -> str:
    if not results:
        return "Please ask a question so I can look it up in the official FAQs."

    found_results = [result for result in results if result.found and result.hit is not None]
    missing_results = [result for result in results if not result.found]

    if not found_results:
        if support_contact_line:
            return f"{NOT_FOUND_MESSAGE}\n\nFor further details, please contact {support_contact_line}."
        return NOT_FOUND_MESSAGE

    if len(results) == 1 and len(found_results) == 1 and not missing_results:
        return f"The information I found is:\n\n{format_answer_text(found_results[0].hit.entry.answer)}"

    bullets = [
        f"- {result.hit.entry.question}: {format_answer_text(result.hit.entry.answer)}"
        for result in found_results
        if result.hit is not None
    ]

    if missing_results and found_results:
        missing_topics = ", ".join(result.sub_query for result in missing_results)
        message = (
            f"I could not find information about {missing_topics}, but here is what I found:\n\n"
            + "\n".join(bullets)
        )
        if support_contact_line:
            message += f"\n\nFor further details, please contact {support_contact_line}."
        return message

    return "Here's what I found:\n\n" + "\n".join(bullets)

