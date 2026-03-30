import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


FAQ_INPUT_CANDIDATES = [
    Path("data/nust_faq.json"),
    Path("../approach2/data/nust_faq.json"),
    Path("../data/nust_faq.json"),
]
FAQ_OUTPUT = Path("data/nust_faq_enriched.json")
META_OUTPUT = Path("data/link_scrape_cache.json")

URL_RE = re.compile(r"https?://[^\s)]+")

STOPWORDS = {
    "the", "is", "a", "an", "and", "or", "to", "for", "of", "in", "on", "at", "with", "from",
    "what", "how", "can", "i", "if", "do", "does", "are", "be", "my", "your", "it", "this",
    "that", "by", "as", "will", "would", "should", "about", "into", "when", "where", "which",
}


def resolve_faq_path() -> Path:
    for p in FAQ_INPUT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find source nust_faq.json")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def clean_url(url: str) -> str:
    u = url.strip().rstrip(".,;)")
    u = u.replace("InKarachi", "").replace("InQuetta", "")

    # Fix malformed short links created by scrape artifacts.
    if "rb.gy/" in u:
        parsed = urlparse(u)
        code = parsed.path.lstrip("/")
        if "In" in code:
            code = code.split("In", 1)[0]
        if len(code) > 8:
            code = code[:8]
        if code:
            u = f"{parsed.scheme}://{parsed.netloc}/{code}"

    # Remove duplicated slashes in path while preserving scheme.
    u = re.sub(r"(?<!:)//+", "/", u)
    if u.startswith("http:/") and not u.startswith("http://"):
        u = u.replace("http:/", "http://", 1)
    if u.startswith("https:/") and not u.startswith("https://"):
        u = u.replace("https:/", "https://", 1)
    return u


def extract_urls(answer: str):
    urls = [clean_url(u) for u in URL_RE.findall(answer)]
    # Dedup preserving order.
    out = []
    seen = set()
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def scrape_url(url: str, timeout: int = 20) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) NUST-Offline-Builder/1.0"
    }
    r = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # Remove noisy blocks.
    for tag in soup(["script", "style", "noscript", "svg", "footer", "header", "nav", "form"]):
        tag.decompose()

    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text


def split_sentences(text: str):
    # basic sentence split
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 25]


def score_sentence(question: str, sent: str) -> float:
    q_tokens = [t for t in re.findall(r"[a-z0-9]+", normalize_text(question)) if t not in STOPWORDS]
    s_norm = normalize_text(sent)

    if not q_tokens:
        return 0.0

    overlap = sum(1 for t in q_tokens if t in s_norm)
    score = overlap / max(len(set(q_tokens)), 1)

    # Bonus for numeric/fee detail lines.
    if any(k in s_norm for k in ["rs", "usd", "fee", "net", "deadline", "eligibility", "deposit"]):
        score += 0.12
    # Penalty for cookie/policy junk.
    if any(k in s_norm for k in ["cookie", "privacy", "copyright", "facebook", "twitter"]):
        score -= 0.20
    return score


def summarize_for_question(question: str, page_texts):
    all_sentences = []
    for txt in page_texts:
        all_sentences.extend(split_sentences(txt))

    if not all_sentences:
        return ""

    scored = [(score_sentence(question, s), s) for s in all_sentences]
    scored.sort(key=lambda x: x[0], reverse=True)

    picked = []
    seen = set()
    for sc, s in scored:
        if sc <= 0:
            continue
        key = normalize_text(s)
        if key in seen:
            continue
        seen.add(key)
        picked.append(s)
        if len(picked) == 3:
            break

    if not picked:
        return ""

    summary = " ".join(picked)
    # Keep concise for app responses.
    if len(summary) > 650:
        summary = summary[:650].rsplit(" ", 1)[0] + "..."
    return summary


def main():
    faq_path = resolve_faq_path()
    data = json.loads(faq_path.read_text(encoding="utf-8"))

    # Gather all unique links.
    all_links = []
    for e in data:
        all_links.extend(extract_urls(str(e.get("answer", ""))))

    unique_links = []
    seen = set()
    for u in all_links:
        if u not in seen:
            seen.add(u)
            unique_links.append(u)

    print(f"FAQ path: {faq_path}")
    print(f"Entries: {len(data)}")
    print(f"Unique links: {len(unique_links)}")

    # Scrape each link once.
    link_cache = {}
    failed = {}
    for i, u in enumerate(unique_links, start=1):
        try:
            txt = scrape_url(u)
            link_cache[u] = txt
            print(f"[{i}/{len(unique_links)}] OK  {u}  ({len(txt)} chars)")
        except Exception as exc:
            failed[u] = str(exc)
            print(f"[{i}/{len(unique_links)}] ERR {u} :: {exc}")
        time.sleep(0.2)

    # Build enriched entries.
    enriched = []
    enriched_count = 0
    for e in data:
        q = str(e.get("question", "")).strip()
        a = str(e.get("answer", "")).strip()
        urls = extract_urls(a)

        if not urls:
            enriched.append({"question": q, "answer": a, "links": [], "answer_mode": "original"})
            continue

        texts = [link_cache[u] for u in urls if u in link_cache]
        summary = summarize_for_question(q, texts)
        if summary:
            enriched_answer = summary
            mode = "scraped_summary"
            enriched_count += 1
        else:
            # Fallback to original when scraping fails.
            enriched_answer = a
            mode = "original_fallback"

        enriched.append({
            "question": q,
            "answer": enriched_answer,
            "links": urls,
            "answer_mode": mode,
            "original_answer": a,
        })

    FAQ_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    FAQ_OUTPUT.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")

    META_OUTPUT.write_text(
        json.dumps(
            {
                "source_faq": str(faq_path),
                "unique_links": len(unique_links),
                "scraped_ok": len(link_cache),
                "scraped_failed": len(failed),
                "failed_links": failed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nDone")
    print(f"Enriched answers: {enriched_count}")
    print(f"Output: {FAQ_OUTPUT}")
    print(f"Meta: {META_OUTPUT}")


if __name__ == "__main__":
    main()
