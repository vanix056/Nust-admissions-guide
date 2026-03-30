import json
import os
import re
import time
import html
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

# Keep offline behavior deterministic.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import streamlit as st
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer
import faiss


# --------------------------
# Config
# --------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_CANDIDATES = [
    "phi-3-mini-4k-instruct-q4_K_M.gguf",
    "Phi-3-mini-4k-instruct-q4.gguf",
]
TOP_K = 6
MIN_SEMANTIC_CONF = 0.34
MIN_FUZZY_CONF = 62

# Speed controls
FAST_RETURN_SEMANTIC = 0.66
FAST_RETURN_FUZZY = 84
LLM_TOP_K = 3
LLM_MAX_TOKENS = 120
LLM_CTX_SIZE = 2048
LLM_LATENCY_TARGET_MS = 5000

# In LLM mode, bypass generation for high-confidence retrieval hits.
LLM_BYPASS_SEMANTIC = 0.82
LLM_BYPASS_FUZZY = 92

# Performance knobs (all are opt-in so LLM mode remains truly generative by default).
# Set LLM_CPU_FAST_MODE=1 to force retrieval in LLM mode.
LLM_CPU_FAST_MODE = os.getenv("LLM_CPU_FAST_MODE", "0") == "1"
# Set LLM_BYPASS_IN_LLM_MODE=1 to allow retrieval bypass at very high confidence.
LLM_BYPASS_IN_LLM_MODE = os.getenv("LLM_BYPASS_IN_LLM_MODE", "0") == "1"
# Set LLM_ENFORCE_LATENCY_BUDGET=1 to force retrieval fallback when generation exceeds target latency.
LLM_ENFORCE_LATENCY_BUDGET = os.getenv("LLM_ENFORCE_LATENCY_BUDGET", "0") == "1"

UNKNOWN_MESSAGE = (
    "I could not find a reliable answer in the official NUST FAQ knowledge base. "
    "Please try rephrasing your question.\n\n"
    "For further guidance, contact NUST Undergraduate Admissions: "
    "https://nust.edu.pk/admissions/"
)
URL_RE = re.compile(r"https?://[^\s)]+")


# --------------------------
# Utilities
# --------------------------
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def clean_url(url: str) -> str:
    u = url.strip().rstrip(".,;)")
    u = u.replace("InKarachi", "").replace("InQuetta", "")
    if "rb.gy/" in u:
        parsed = urlparse(u)
        code = parsed.path.lstrip("/")
        if "In" in code:
            code = code.split("In", 1)[0]
        if len(code) > 8:
            code = code[:8]
        if code:
            u = f"{parsed.scheme}://{parsed.netloc}/{code}"
    return u


def extract_urls(text: str) -> List[str]:
    text = re.sub(r"\.\s+(aspx\b)", r".\1", text or "", flags=re.IGNORECASE)
    urls = [clean_url(u) for u in URL_RE.findall(text)]
    out = []
    seen = set()
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def strip_urls(text: str) -> str:
    normalized = re.sub(r"\.\s+(aspx\b)", r".\1", text or "", flags=re.IGNORECASE)
    stripped = URL_RE.sub("", normalized)
    stripped = re.sub(r"\b(aspx|php|html?)\b", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"(?:in the )?following link:?\s*$", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped


def _link_lookup_variants(url: str) -> List[str]:
    u = clean_url(url)
    variants = [u]
    if u.endswith("/"):
        variants.append(u.rstrip("/"))
    else:
        variants.append(u + "/")
    if u.startswith("https://"):
        variants.append("http://" + u[len("https://"):])
    if u.startswith("http://"):
        variants.append("https://" + u[len("http://"):])
    seen = set()
    out = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


@st.cache_resource(show_spinner=False)
def load_offline_link_knowledge() -> Dict[str, str]:
    candidates = [
        Path("data/link_offline_knowledge.json"),
        Path("../LLM/data/link_offline_knowledge.json"),
    ]
    for p in candidates:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                return {clean_url(str(k)): str(v).strip() for k, v in raw.items() if str(v).strip()}
    return {}


def build_offline_answer_from_links(question: str, answer: str, link_knowledge: Dict[str, str]) -> str:
    urls = extract_urls(answer)
    if not urls:
        return answer

    snippets = []
    for u in urls:
        for key in _link_lookup_variants(u):
            if key in link_knowledge:
                snippets.append(link_knowledge[key])
                break

    base_answer = strip_urls(answer)
    if not snippets:
        return base_answer or answer

    # Prefer concise and question-relevant snippets.
    q_terms = [t for t in re.findall(r"[a-z0-9]+", normalize_text(question)) if len(t) > 2]

    def score(snippet: str) -> int:
        s = normalize_text(snippet)
        return sum(1 for t in q_terms if t in s)

    snippets = sorted(snippets, key=score, reverse=True)
    selected = []
    seen = set()
    for s in snippets:
        k = normalize_text(s)
        if k in seen:
            continue
        seen.add(k)
        selected.append(s)
        if len(selected) >= 2:
            break

    link_context = " ".join(selected).strip()
    if base_answer and link_context:
        return f"{base_answer} {link_context}".strip()
    return base_answer or link_context or answer


def _collapse_repeated_letters(word: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", word)


def detect_small_talk(query: str) -> Tuple[str, str]:
    q = normalize_text(query)

    greetings = {"hi", "hello", "hey", "salam", "assalamualaikum", "aoa"}
    if q in greetings or any(q.startswith(g + " ") for g in greetings):
        return (
            "Hello. I am your NUST admissions assistant. Ask any admissions question and I will answer using the official FAQ knowledge base.",
            "small_talk",
        )

    if "how are you" in q:
        return ("I am doing well, thank you. I am ready to help with NUST admissions questions.", "small_talk")

    if any(w in q for w in ["thanks", "thank you", "jazakallah", "shukriya"]):
        return ("You are welcome. If you need more help with admissions, ask me anything.", "small_talk")

    if any(w in q for w in ["bye", "goodbye", "see you", "allah hafiz"]):
        return ("Goodbye. Wishing you the best for your admissions journey.", "small_talk")

    return "", ""


def format_answer_text(answer: str) -> str:
    text = answer.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    fixes = {
        "InIslamabad": "In Islamabad",
        "InQuetta": "In Quetta",
        "InKarachi": "In Karachi",
        "andGilgit": "and Gilgit",
        "hoursand": "hours and",
        "theMDCATconductedbyNUMS": "the MDCAT conducted by NUMS",
        "programsoffered": "programs offered",
        "under theFee": "under the Fee",
        "Bank)ATM": "Bank), ATM",
        "Bank)Easy": "Bank), Easy",
        "SeatsNET": "Seats NET",
        "DAEThe": "DAE. The",
        "Location: and Gilgit": "Location. In Karachi and Gilgit",
        "Pin location: NET is conducted": "In Quetta, NET is conducted",
       "www.ugadmissions.nust.edu.pkunder": "www.ugadmissions.nust.edu.pk under",
       "policyclick": "policy, click",
       "detailclick": "detail, click",
       "criteriaclick": "criteria, click",
        "linkSample": "link Sample",
       ".under the": ". Under the",
    }
    for bad, good in fixes.items():
        text = text.replace(bad, good)
    text = re.sub(r"Please visit:\s*", "Please visit: ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(the following links?|following link)\b\s*:?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(Please visit:\s*https?://[^\s]+)(?:\s+\1)+", r"\1", text, flags=re.IGNORECASE)
    if not re.search(r"https?://|\bwww\.", text, flags=re.IGNORECASE):
        text = re.sub(r"\bclick here\b\.?", "", text, flags=re.IGNORECASE).strip()

    # Collapse duplicate sentence blocks if model repeats itself.
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    if parts:
        dedup_parts = []
        seen = set()
        for p in parts:
            key = normalize_text(p)
            if key not in seen:
                seen.add(key)
                dedup_parts.append(p)
        text = " ".join(dedup_parts)

    return text


def conversationalize_answer(query: str, answer: str) -> str:
    text = format_answer_text(answer)
    if not text:
        return text

    qn = normalize_text(query)
    if qn.startswith(("hi", "hello", "hey", "salam", "aoa")):
        return text

    if re.match(r"^(yes|no|you can|there is|result|admissions|for)\b", text, re.IGNORECASE):
        return text
    return f"Based on the official FAQ, {text}"


def append_links(answer: str, links: List[str]) -> str:
    if not links:
        return format_answer_text(answer)
    cleaned = []
    seen = set()
    for url in links:
        u = clean_url(str(url))
        if u and u not in seen:
            seen.add(u)
            cleaned.append(u)
    if not cleaned:
        return format_answer_text(answer)
    return f"{format_answer_text(answer)}\n\nOfficial link(s):\n" + "\n".join(cleaned)


def make_urls_clickable(text: str) -> str:
    # Convert plain URLs and www-domains into markdown links.
    if not text:
        return text

    placeholders: List[str] = []

    def protect_md_links(match):
        placeholders.append(match.group(0))
        return f"__MDLINK_{len(placeholders)-1}__"

    protected = re.sub(r"\[[^\]]+\]\(https?://[^)]+\)", protect_md_links, text)

    protected = re.sub(
        r"(?<!\()https?://[^\s)]+",
        lambda m: f"[{m.group(0)}]({m.group(0)})",
        protected,
    )
    protected = re.sub(
        r"\bwww\.[^\s)]+",
        lambda m: f"[{m.group(0)}](https://{m.group(0)})",
        protected,
    )

    for i, link in enumerate(placeholders):
        protected = protected.replace(f"__MDLINK_{i}__", link)

    return protected

def embed_links_inline(answer: str, links: List[str]) -> str:
    """Embed links inline using markdown format [Entity](url) where available."""
    if not links or not answer:
        formatted = format_answer_text(answer)
        return make_urls_clickable(formatted)
    
    formatted = format_answer_text(answer)
    cleaned_links = []
    seen = set()
    for url in links:
        u = clean_url(str(url))
        if u and u not in seen:
            seen.add(u)
            cleaned_links.append(u)
    
    if not cleaned_links:
        return make_urls_clickable(formatted)
    
    result = formatted
    
    # Try to embed up to 2 links inline for key entities
    entity_replacements = [
        (r'\bNUST Exam Hall\b', cleaned_links[0] if len(cleaned_links) > 0 else None),
        (r'\bNUST Balochistan Campus\b|\bNBC\b', cleaned_links[1] if len(cleaned_links) > 1 else cleaned_links[0] if cleaned_links else None),
        (r'\bwww\.ugadmissions\.nust\.edu\.pk\b', cleaned_links[0] if cleaned_links else None),
    ]
    
    embedded_count = 0
    for pattern, url in entity_replacements:
        if url is None:
            continue
        def replacer(match):
            nonlocal embedded_count
            embedded_count += 1
            return f"[{match.group(0)}]({url})"
        result = re.sub(pattern, replacer, result, count=1)
    
    # Append any remaining links at end
    remaining_links = cleaned_links[embedded_count:]
    if remaining_links:
        result = f"{result}\n\nOfficial link(s):\n" + "\n".join(remaining_links)

    return make_urls_clickable(result)

def score_faq_relevance(query: str, question: str, semantic_score: float) -> float:
    q_terms = set(normalize_text(query).split())
    faq_terms = set(normalize_text(question).split())

    overlap = len(q_terms.intersection(faq_terms)) / max(len(q_terms), 1)

    specific_keywords = {"bshnd", "mbbs", "nshs", "dae", "o level", "a level"}
    general_keywords = {"different", "ug", "undergraduate", "programmes", "programs"}
    has_specific = len(faq_terms.intersection(specific_keywords)) > 0
    has_general = len(faq_terms.intersection(general_keywords)) > 0

    specificity_penalty = 0.0
    if has_specific and not any(kw in normalize_text(query) for kw in specific_keywords):
        specificity_penalty = 0.15

    general_bonus = 0.0
    if has_general and not any(kw in normalize_text(query) for kw in specific_keywords):
        general_bonus = 0.10

    combined = semantic_score + (0.1 * overlap) - specificity_penalty + general_bonus
    return max(0, combined)


def _is_general_fee_structure_intent(query_norm: str) -> bool:
    """Detect generic fee-structure queries (not NET/refund-specific)."""
    tokens = set(re.findall(r"[a-z0-9]+", query_norm))
    if "fee" not in tokens:
        return False

    specific_tokens = {
        "net",
        "entry",
        "test",
        "refundable",
        "refund",
        "application",
        "processing",
        "security",
        "deposit",
        "selected",
    }

    if tokens.intersection(specific_tokens):
        return False

    return ("structure" in tokens) or (len(tokens) <= 3)


def _general_fee_faq_index(questions: List[str], answers: List[str]) -> int:
    for i, answer in enumerate(answers):
        if "undergraduate-financial-matters" in answer.lower():
            return i

    for i, question in enumerate(questions):
        q = normalize_text(question)
        if "fee structure" in q and ("different" in q or "ug" in q or "undergraduate" in q):
            return i

    return -1


def _intent_override_index(query_norm: str, questions: List[str], answers: List[str]) -> int:
    """Route clear intents to exact FAQ entries for precision and speed."""
    tokens = set(re.findall(r"[a-z0-9]+", query_norm))

    def find_best(match_fn) -> int:
        for i, (q, a) in enumerate(zip(questions, answers)):
            if match_fn(normalize_text(q), normalize_text(a)):
                return i
        return -1

    # Refundability of admission/application processing fee.
    if "fee" in tokens and tokens.intersection({"refund", "refundable", "selected", "join"}):
        idx = find_best(lambda q, a: "refundable" in q and "admission processing fee" in q)
        if idx >= 0:
            return idx

    # Security deposit / processing fee refund on not joining.
    if tokens.intersection({"admission", "join", "money", "refund", "refundable", "back"}) and (
        ("fee" in tokens) or ("deposit" in tokens) or ("security" in tokens)
    ):
        idx = find_best(
            lambda q, a: "security deposit" in q and "does not join the university" in q
        )
        if idx >= 0:
            return idx

    # Refund question phrased without explicit "fee/deposit" words.
    if tokens.intersection({"admission", "join"}) and tokens.intersection({"money", "back", "refund", "refundable"}):
        idx = find_best(
            lambda q, a: "security deposit" in q and "does not join the university" in q
        )
        if idx >= 0:
            return idx

    # NET fee exemption for HSSC board position holders.
    if "net" in tokens and "fee" in tokens and tokens.intersection({"first", "second", "third", "position", "stood", "board", "hssc", "exempted"}):
        idx = find_best(lambda q, a: "exempted from payment of net application processing fee" in q)
        if idx >= 0:
            return idx

    # NET negative marking / penalty / deduction.
    if "net" in tokens and tokens.intersection({"negative", "penalty", "deduction", "wrong", "marking", "answers"}):
        idx = find_best(lambda q, a: "negative marking" in q and "entry test" in q)
        if idx >= 0:
            return idx

    # General quota / reserved seats (non-MBBS specific).
    if tokens.intersection({"quota", "reserved"}) and not tokens.intersection({"mbbs", "nshs"}):
        idx = find_best(lambda q, a: "are there any quota / reserved seats?" in q)
        if idx >= 0:
            return idx

    # Fee revision/stability query (year to year).
    if "fee" in tokens and tokens.intersection({"remain", "same", "throughout", "degree", "first", "year", "revised", "revised", "change", "changed"}):
        idx = find_best(lambda q, a: "how frequently are university fee rates revised" in q)
        if idx >= 0:
            return idx

    # NET fee for Pakistani nationals.
    if "fee" in tokens and "net" in tokens:
        idx = find_best(
            lambda q, a: "application processing fee" in q
            and "nust entry test" in a
            and ("pakistani" in a or "pakistani" in q)
        )
        if idx >= 0:
            return idx

    # SAT/ACT institutional code query should route to score-submission deadline FAQ.
    if tokens.intersection({"sat", "act"}) and tokens.intersection({"institutional", "code", "scores", "score"}):
        idx = find_best(lambda q, a: "deadline for submission of act / sat score" in q)
        if idx >= 0:
            return idx

    # Online fee payment methods (1Link/card/banking) should not route to fee amount.
    if "fee" in tokens and (
        "online" in tokens
        or bool(tokens.intersection({"1link", "bill", "invoice", "card", "banking", "easypaisa", "jazzcash"}))
        or ("payment" in tokens and "pay" in tokens)
    ):
        idx = find_best(
            lambda q, a: (
                "submit the application processing fee (online)" in q
                or (
                    "submit the application processing fee" in q
                    and any(x in a for x in ["1 link", "credit card", "online banking", "easy paisa", "jazz cash"])
                )
            )
        )
        if idx >= 0:
            return idx

    # ICS background applying for engineering.
    if "ics" in tokens and "engineering" in tokens:
        idx = find_best(lambda q, a: "ics" in q and "engineering" in q)
        if idx >= 0:
            return idx

    # Generic engineering start/eligibility/program options query.
    if "engineering" in tokens and tokens.intersection({"study", "start", "begin", "where", "apply", "programmes", "programs", "options"}):
        idx = find_best(
            lambda q, a: "various academics backgrounds" in q and "ug disciplines" in q
        )
        if idx >= 0:
            return idx

    # Missed NET session / reschedule intent.
    if "net" in tokens and tokens.intersection({"miss", "missed", "reschedule", "session", "other", "day"}):
        idx = find_best(lambda q, a: "could not appear" in q and "entry test" in q)
        if idx >= 0:
            return idx

    # MCQs + duration combined NET question.
    if "net" in tokens and tokens.intersection({"mcq", "mcqs", "duration", "long", "hours", "time"}):
        idx = find_best(lambda q, a: "duration of test and the number of mcqs" in q)
        if idx >= 0:
            return idx

    # NET subjects/weightings query.
    if "net" in tokens and ("weightings" in tokens or "weighting" in tokens or "subjects" in tokens):
        idx = find_best(
            lambda q, a: (
                "subjects included in net with weightings" in q
                or "subjects for entry test" in q
                or "syllabus of entry test subjects" in q
            )
        )
        if idx >= 0:
            return idx

    # NET result announcement timeline.
    if "net" in tokens and tokens.intersection({"result", "announcement", "timeline", "when", "soon", "announced"}):
        idx = find_best(lambda q, a: "timeline" in q and "net result" in q)
        if idx >= 0:
            return idx

    # NET entry test syllabus / format / subjects.
    if ("net" in tokens or "nust" in tokens or "entry" in tokens) and tokens.intersection({"syllabus", "format", "curriculum", "covered"}):
        idx = find_best(lambda q, a: "syllabus of entry test" in q and "format" in q)
        if idx >= 0:
            return idx

    # Foreigner/international applicant intent.
    if tokens.intersection({"foreigner", "foreign", "international"}) and tokens.intersection({"apply", "admission", "nust"}):
        idx = find_best(lambda q, a: "foreigner" in q and "admission" in q and "nust" in q)
        if idx >= 0:
            return idx
        idx = find_best(
            lambda q, a: (
                "international students" in q
                or "international students" in a
                or ("international seat" in a and "expatriate" not in q)
            )
        )
        if idx >= 0:
            return idx

    # Expatriate/dual-nationality category and eligibility queries.
    if tokens.intersection({"dual", "nationality", "passport", "expatriate", "local", "international", "abroad", "uk"}) and tokens.intersection({"apply", "category", "student", "test", "need", "can"}):
        if tokens.intersection({"test", "need", "apply", "which"}):
            idx = find_best(lambda q, a: "how can i apply at nust if i am an expatriate student" in q)
            if idx >= 0:
                return idx
        idx = find_best(lambda q, a: "do i fall under the expatriate students category" in q)
        if idx >= 0:
            return idx
        idx = find_best(lambda q, a: "born pakistani with a foreign passport" in q)
        if idx >= 0:
            return idx

    # NSHS programme count besides MBBS.
    if tokens.intersection({"nshs", "programme", "programmes", "program", "offer", "offerings"}) and tokens.intersection({"how", "many", "besides", "mbbs"}):
        idx = find_best(lambda q, a: "how many allied programmes does nshs offer" in q)
        if idx >= 0:
            return idx

    # Hostel availability for MBBS students (boys/girls).
    if tokens.intersection({"hostel", "girls", "girl", "mbbs", "available", "facility"}):
        idx = find_best(lambda q, a: "hostel facility" in q and "mbbs" in q)
        if idx >= 0:
            return idx

    return -1


def _resolve_faq_path() -> Path:
    candidates = [
        Path("data/nust_faq_enriched.json"),
        Path("data/nust_faq.json"),
        Path("../data/nust_faq.json"),
        Path("../approach2/data/nust_faq.json"),
        Path("../approach 1/data/nust_faq.json"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not locate nust_faq.json in known locations.")


def _resolve_model_path() -> Path:
    custom = os.getenv("LLM_MODEL_PATH", "").strip()
    if custom and Path(custom).exists():
        return Path(custom)

    candidates: List[Path] = []
    for name in MODEL_CANDIDATES:
        candidates.append(Path(name))
        candidates.append(Path("../approach2") / name)

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "No GGUF model found. Place a Phi-3 GGUF file in LLM/ or approach2/, "
        "or set LLM_MODEL_PATH environment variable."
    )


@st.cache_resource(show_spinner=False)
def response_cache() -> Dict[Tuple[str, bool], Tuple[str, Dict[str, str]]]:
    """Simple in-memory cache for repeated normalized queries."""
    return {}


@st.cache_resource(show_spinner=False)
def load_faq_data() -> Tuple[List[Dict[str, Any]], List[str], List[str], Path]:
    faq_path = _resolve_faq_path()
    link_knowledge = load_offline_link_knowledge()
    with faq_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("FAQ JSON must be a list of objects.")

    entries: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        a = build_offline_answer_from_links(q, a, link_knowledge)
        links = item.get("links", [])
        if not isinstance(links, list):
            links = []
        if q and a:
            entries.append({"question": q, "answer": a, "links": links})

    if not entries:
        raise ValueError("No valid FAQ entries found.")

    questions = [x["question"] for x in entries]
    answers = [x["answer"] for x in entries]
    return entries, questions, answers, faq_path


@st.cache_resource(show_spinner=False)
def build_question_vocab(questions: List[str]) -> List[str]:
    vocab = set()
    for q in questions:
        for tok in re.findall(r"[a-z0-9]+", normalize_text(q)):
            if len(tok) >= 2:
                vocab.add(tok)
    return sorted(vocab)


def normalize_query_for_matching(query: str, vocab: List[str]) -> str:
    alias_map = {
        "nsut": "nust",
        "nuust": "nust",
        "nusst": "nust",
        "nustt": "nust",
        "wht": "what",
        "whaat": "what",
        "wat": "what",
        "iz": "is",
        "fe": "fee",
        "fees": "fee",
        "strctr": "structure",
        "strctre": "structure",
        "strcture": "structure",
        "strucure": "structure",
    }

    raw_tokens = re.findall(r"[a-z0-9]+", normalize_text(query))
    corrected: List[str] = []

    for token in raw_tokens:
        token = _collapse_repeated_letters(token)
        token = alias_map.get(token, token)

        if token in vocab or len(token) <= 2 or token.isdigit():
            corrected.append(token)
            continue

        best = process.extractOne(token, vocab, scorer=fuzz.ratio)
        if best:
            word, score, _ = best
            if score >= 86 and abs(len(word) - len(token)) <= 3:
                corrected.append(word)
                continue

        corrected.append(token)

    return " ".join(corrected).strip() or normalize_text(query)


@st.cache_resource(show_spinner=False)
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu", local_files_only=True)


@st.cache_resource(show_spinner=False)
def build_index(questions: List[str]) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    model = load_embedder()
    vectors = model.encode(
        questions,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
    ).astype(np.float32)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, vectors


@st.cache_resource(show_spinner=False)
def load_llm():
    from llama_cpp import Llama

    model_path = _resolve_model_path()
    # CPU-only default tuned for mid-range laptops (e.g., i5 without GPU).
    cpu_threads = int(os.getenv("LLM_THREADS", str(max(4, min(12, (os.cpu_count() or 8))))))
    # On macOS builds with Metal enabled, set LLM_N_GPU_LAYERS=-1 for significant speedups.
    n_gpu_layers = int(os.getenv("LLM_N_GPU_LAYERS", "0"))
    return Llama(
        model_path=str(model_path),
        n_ctx=LLM_CTX_SIZE,
        n_threads=cpu_threads,
        n_batch=256,
        n_gpu_layers=n_gpu_layers,
        f16_kv=True,
        temperature=0,
        verbose=False,
    )


def retrieve_candidate_indices(
    query_for_match: str,
    index: faiss.IndexFlatIP,
    questions: List[str],
    answers: List[str],
) -> Tuple[List[int], float, float]:
    model = load_embedder()
    q_vec = model.encode([query_for_match], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_vec)

    sem_scores, sem_indices = index.search(q_vec, TOP_K)
    best_sem = float(sem_scores[0][0])

    sem_map: Dict[int, float] = {}
    candidates = []
    for rank, idx in enumerate(sem_indices[0]):
        i = int(idx)
        if i < 0:
            continue
        candidates.append(i)
        sem_map[i] = float(sem_scores[0][rank])

    # Add best fuzzy candidate to make typo-heavy queries robust.
    fuzzy_scored = []
    qn = normalize_text(query_for_match)
    for i, q in enumerate(questions):
        q_norm = normalize_text(q)
        score = max(
            fuzz.token_set_ratio(qn, q_norm),
            fuzz.token_sort_ratio(qn, q_norm),
            fuzz.partial_ratio(qn, q_norm),
            fuzz.ratio(qn, q_norm),
        )
        fuzzy_scored.append((score, i))
    fuzzy_scored.sort(reverse=True)
    best_fuzzy, best_fuzzy_idx = fuzzy_scored[0]

    fuzzy_map: Dict[int, float] = {best_fuzzy_idx: best_fuzzy / 100.0}
    if best_fuzzy >= MIN_FUZZY_CONF:
        candidates.append(best_fuzzy_idx)

    # Deduplicate preserving order.
    dedup = []
    seen = set()
    for idx in candidates:
        if idx not in seen:
            seen.add(idx)
            dedup.append(idx)

    # Rank candidates with semantic + fuzzy + FAQ relevance score.
    ranked = []
    for idx in dedup:
        sem_norm = sem_map.get(idx, 0.0)
        fuzzy_norm = fuzzy_map.get(idx, 0.0)
        qn = normalize_text(questions[idx])
        rel = score_faq_relevance(query_for_match, questions[idx], sem_norm)
        combined = (0.65 * sem_norm) + (0.35 * fuzzy_norm) + (0.20 * rel)

        # Extra lexical boosts for short definition queries (e.g., "what is net").
        if query_for_match and query_for_match in qn:
            combined += 0.30
        query_tokens = [t for t in re.findall(r"[a-z0-9]+", query_for_match) if t not in {"what", "is", "the", "a", "an"}]
        if query_tokens and all(t in qn for t in query_tokens):
            combined += 0.12
        if "what is net" in query_for_match and ("what is net" in qn or "net" in qn):
            combined += 0.22

        ranked.append((combined, idx))
    ranked.sort(reverse=True)

    ordered = [idx for _, idx in ranked]

    # Fee-intent safety: include general fee FAQ at front for generic fee questions.
    if _is_general_fee_structure_intent(query_for_match):
        fee_idx = _general_fee_faq_index(questions, answers)
        if fee_idx >= 0 and not any(k in query_for_match for k in ["bshnd", "mbbs", "nshs"]):
            ordered = [fee_idx] + [i for i in ordered if i != fee_idx]

    return ordered, best_sem, float(best_fuzzy)


def build_grounded_prompt(query: str, candidate_indices: List[int], entries: List[Dict[str, Any]]) -> str:
    context_parts = []
    for rank, idx in enumerate(candidate_indices[:LLM_TOP_K], start=1):
        item = entries[idx]
        # Keep context focused but sufficiently rich for synthesis.
        short_answer = format_answer_text(item["answer"])[:900]
        context_parts.append(f"[{rank}] Q: {item['question']}\n[{rank}] A: {short_answer}")

    context = "\n\n".join(context_parts)
    return (
        "You are a careful NUST admissions assistant. "
        "Use only the FAQ context.\n\n"
        f"FAQ Context:\n{context}\n\n"
        f"Student Question: {query}\n\n"
        "Instructions:\n"
        "1) Prefer item [1], but synthesize across [2]/[3] when needed.\n"
        "2) Provide a clear, direct answer in 2-4 sentences.\n"
        "3) If asked to compare or choose, explicitly contrast options from context.\n"
        "4) Include no invented links.\n\n"
        "Final Answer:"
    )


def _strip_generation_artifacts(text: str) -> str:
    cleaned = text
    cleaned = cleaned.replace("<|assistant|>", "")
    cleaned = cleaned.replace("Solution:", "")
    cleaned = cleaned.replace("Final Answer:", "")
    cleaned = re.sub(r"==[^=]+==", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_links(text: str) -> List[str]:
    return re.findall(r"https?://[^\s)]+", text)


def split_compound_query(query: str) -> List[str]:
    """Split multi-part/numbered user input into separate questions."""
    if not query or not query.strip():
        return []

    text = query.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"\?{2,}", "?", text)

    # If user pasted extra content around a quoted question, prioritize the quoted question.
    quoted_q = re.search(r'"\s*([^"\n]{4,200}\?)\s*"', text)
    if quoted_q:
        return [quoted_q.group(1).strip()]

    # Case 1: multi-line with numbering like "6. ..."
    if "\n" in text:
        parts = []
        for line in [ln.strip() for ln in text.split("\n") if ln.strip()]:
            cleaned = re.sub(r"^\d+[\)\.\-:\s]*", "", line).strip()
            qn = normalize_text(cleaned)
            is_question_like = ("?" in cleaned) or bool(re.match(r"^(what|how|is|are|can|do|does|did|will|where|when|which|who|whom|why|i|my)\b", qn))
            if cleaned and is_question_like:
                parts.append(cleaned)
        if len(parts) > 1:
            return parts

    # Case 2: one line containing multiple questions.
    question_chunks = [p.strip() for p in re.split(r"\?\s*", text) if p.strip()]
    if len(question_chunks) > 1:
        filtered = []
        for chunk in question_chunks:
            qn = normalize_text(chunk)
            is_question_like = bool(re.match(r"^(what|how|is|are|can|do|does|did|will|where|when|which|who|whom|why|i|my)\b", qn))
            if is_question_like:
                filtered.append(chunk if chunk.endswith("?") else f"{chunk}?")
        if len(filtered) > 1:
            return filtered
        if len(filtered) == 1:
            return filtered

    # Case 3: conjunction queries like "how many mcqs ... and how long ...".
    lowered = normalize_text(text)
    if " and " in lowered and any(k in lowered for k in ["how many", "how long", "duration", "mcq", "mcqs"]):
        parts = [p.strip() for p in re.split(r"\band\b", text, flags=re.IGNORECASE) if p.strip()]
        if len(parts) > 1:
            normalized_parts = []
            for p in parts:
                if not p.endswith("?"):
                    p = f"{p}?"
                normalized_parts.append(p)
            return normalized_parts

    # Case 4: explicit two-intent query for quota + negative marking.
    if " and " in lowered and any(k in lowered for k in ["quota", "reserved"]) and any(k in lowered for k in ["negative", "wrong answers", "marking", "deduction", "penalty"]):
        return [
            "Are there any quota / reserved seats?",
            "Is there any negative marking in the NUST Entry Test?",
        ]

    return [text]


def llm_grounded_answer(query: str, prompt: str, allowed_links: List[str]) -> str:
    model = load_llm()
    out = model(
        prompt,
        max_tokens=LLM_MAX_TOKENS,
        temperature=0.1,
        top_p=0.9,
        top_k=40,
        stop=["\n\n[", "\nQ:", "Student Question:"],
    )
    text = _strip_generation_artifacts(out["choices"][0]["text"].strip())

    if not text:
        return text

    # Guardrail: if model emits links not present in context, reject generation.
    generated_links = _extract_links(text)
    allowed = set(allowed_links)
    if any(link not in allowed for link in generated_links):
        return ""

    return format_answer_text(text)


def _get_single_answer(
    query: str,
    index: faiss.IndexFlatIP,
    entries: List[Dict[str, Any]],
    questions: List[str],
    use_llm_generation: bool = False,
    llm_cpu_fast_mode: bool = LLM_CPU_FAST_MODE,
    llm_bypass_in_llm_mode: bool = LLM_BYPASS_IN_LLM_MODE,
    llm_enforce_latency_budget: bool = LLM_ENFORCE_LATENCY_BUDGET,
) -> Tuple[str, Dict[str, str]]:
    if not query or not query.strip():
        return "Please enter your admissions question first.", {
            "source": "empty_query",
            "confidence": "0.0",
            "matched_question": "",
        }

    small_talk, intent = detect_small_talk(query)
    if intent == "small_talk":
        return small_talk, {
            "source": "small_talk",
            "confidence": "1.0",
            "matched_question": "",
        }

    # Broad request guard: avoid low-value "salient features, click here" style reply.
    q_norm_for_scope = normalize_text(query)
    if any(p in q_norm_for_scope for p in ["tell me everything", "everything about", "all about admissions", "complete admissions info"]):
        return (
            "Based on the official FAQ, admissions can be pursued through NET and/or ACT/SAT routes depending on category and programme. "
            "Key checks include eligibility criteria, accepted test category, fee structure, and schedule deadlines. "
            "Please ask a specific sub-question (for example: eligibility, tests, fee, merit, hostel, or documents) so I can provide the exact official policy answer.",
            {
                "source": "broad_query_guidance",
                "confidence": "1.0",
                "matched_question": "",
            },
        )

    vocab = build_question_vocab(questions)
    query_for_match = normalize_query_for_matching(query, vocab)

    override_idx = _intent_override_index(query_for_match, questions, [e["answer"] for e in entries])
    if override_idx >= 0:
        final = conversationalize_answer(query, entries[override_idx]["answer"])
        final = embed_links_inline(final, entries[override_idx].get("links", []))
        return final, {
            "source": "intent_override",
            "confidence": "intent",
            "matched_question": entries[override_idx]["question"],
        }

    cache_key = (
        query_for_match,
        use_llm_generation,
        llm_cpu_fast_mode,
        llm_bypass_in_llm_mode,
        llm_enforce_latency_budget,
    )
    cached = response_cache().get(cache_key)
    if cached:
        return cached

    candidate_indices, best_sem, best_fuzzy = retrieve_candidate_indices(query_for_match, index, questions, [e["answer"] for e in entries])

    if best_sem < MIN_SEMANTIC_CONF and best_fuzzy < MIN_FUZZY_CONF:
        result = (UNKNOWN_MESSAGE, {
            "source": "unknown",
            "confidence": f"semantic={best_sem:.3f}, fuzzy={best_fuzzy:.1f}",
            "matched_question": "",
        })
        response_cache()[cache_key] = result
        return result

    top_idx = candidate_indices[0] if candidate_indices else 0

    # Hard latency guard for CPU-only environments: skip generation in LLM mode.
    if use_llm_generation and llm_cpu_fast_mode:
        final = conversationalize_answer(query, entries[top_idx]["answer"])
        final = embed_links_inline(final, entries[top_idx].get("links", []))
        result = (
            final,
            {
                "source": "llm_cpu_fast_mode_retrieval",
                "confidence": f"semantic={best_sem:.3f}, fuzzy={best_fuzzy:.1f}",
                "matched_question": entries[top_idx]["question"],
            },
        )
        response_cache()[cache_key] = result
        return result

    # Latency optimization in LLM mode: high-confidence retrieval should skip generation.
    if use_llm_generation and llm_bypass_in_llm_mode and (best_sem >= LLM_BYPASS_SEMANTIC or best_fuzzy >= LLM_BYPASS_FUZZY):
        final = conversationalize_answer(query, entries[top_idx]["answer"])
        final = embed_links_inline(final, entries[top_idx].get("links", []))
        result = (
            final,
            {
                "source": "llm_bypassed_high_confidence",
                "confidence": f"semantic={best_sem:.3f}, fuzzy={best_fuzzy:.1f}",
                "matched_question": entries[top_idx]["question"],
            },
        )
        response_cache()[cache_key] = result
        return result

    # Fast-path: for very confident retrieval, skip LLM generation in Fast mode only.
    if (not use_llm_generation) and (best_sem >= FAST_RETURN_SEMANTIC or best_fuzzy >= FAST_RETURN_FUZZY):
        final = conversationalize_answer(query, entries[top_idx]["answer"])
        final = embed_links_inline(final, entries[top_idx].get("links", []))
        result = (
            final,
            {
                "source": "fast_retrieval",
                "confidence": f"semantic={best_sem:.3f}, fuzzy={best_fuzzy:.1f}",
                "matched_question": entries[top_idx]["question"],
            },
        )
        response_cache()[cache_key] = result
        return result

    # Optional speed mode: skip LLM even on medium-confidence queries.
    if not use_llm_generation:
        final = conversationalize_answer(query, entries[top_idx]["answer"])
        final = embed_links_inline(final, entries[top_idx].get("links", []))
        result = (
            final,
            {
                "source": "retrieval_only",
                "confidence": f"semantic={best_sem:.3f}, fuzzy={best_fuzzy:.1f}",
                "matched_question": entries[top_idx]["question"],
            },
        )
        response_cache()[cache_key] = result
        return result

    prompt = build_grounded_prompt(query, candidate_indices, entries)
    context_answers = [entries[i]["answer"] for i in candidate_indices[:LLM_TOP_K]]
    allowed_links = []
    for ans in context_answers:
        allowed_links.extend(_extract_links(ans))

    start = time.perf_counter()
    try:
        answer = llm_grounded_answer(query, prompt, allowed_links)
    except Exception as exc:
        # Graceful fallback to top retrieved FAQ answer if LLM fails.
        answer = conversationalize_answer(query, entries[top_idx]["answer"])
        answer = embed_links_inline(answer, entries[top_idx].get("links", []))
        result = (answer, {
            "source": "faq_fallback_after_llm_error",
            "confidence": f"semantic={best_sem:.3f}, fuzzy={best_fuzzy:.1f}",
            "matched_question": entries[top_idx]["question"],
            "error": str(exc),
        })
        response_cache()[cache_key] = result
        return result

    # If generation is empty after guardrails, use top grounded FAQ.
    if not answer:
        fallback = conversationalize_answer(query, entries[top_idx]["answer"])
        fallback = embed_links_inline(fallback, entries[top_idx].get("links", []))
        result = (fallback, {
            "source": "faq_fallback_after_guardrail",
            "confidence": f"semantic={best_sem:.3f}, fuzzy={best_fuzzy:.1f}",
            "matched_question": entries[top_idx]["question"],
        })
        response_cache()[cache_key] = result
        return result

    # Additional safety: generic fee queries should not drift to specific programs.
    qn = normalize_text(query_for_match)
    if _is_general_fee_structure_intent(qn) and not any(k in qn for k in ["bshnd", "mbbs", "nshs"]):
        if any(k in normalize_text(answer) for k in ["bshnd", "mbbs", "nshs"]):
            fallback = conversationalize_answer(query, entries[top_idx]["answer"])
            fallback = embed_links_inline(fallback, entries[top_idx].get("links", []))
            result = (fallback, {
                "source": "faq_fallback_after_guardrail",
                "confidence": f"semantic={best_sem:.3f}, fuzzy={best_fuzzy:.1f}",
                "matched_question": entries[top_idx]["question"],
            })
            response_cache()[cache_key] = result
            return result

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    final = conversationalize_answer(query, answer)
    final = embed_links_inline(final, entries[top_idx].get("links", []))
    result = (final, {
        "source": "llm_grounded",
        "confidence": f"semantic={best_sem:.3f}, fuzzy={best_fuzzy:.1f}, llm_ms={elapsed_ms}",
        "matched_question": entries[top_idx]["question"],
    })

    # Optional latency guard: only enforce when explicitly enabled.
    if llm_enforce_latency_budget and elapsed_ms > LLM_LATENCY_TARGET_MS:
        fallback = conversationalize_answer(query, entries[top_idx]["answer"])
        fallback = embed_links_inline(fallback, entries[top_idx].get("links", []))
        result = (
            fallback,
            {
                "source": "faq_fallback_after_latency_budget",
                "confidence": f"semantic={best_sem:.3f}, fuzzy={best_fuzzy:.1f}, llm_ms={elapsed_ms}",
                "matched_question": entries[top_idx]["question"],
            },
        )

    response_cache()[cache_key] = result
    return result


def get_answer(
    query: str,
    index: faiss.IndexFlatIP,
    entries: List[Dict[str, Any]],
    questions: List[str],
    use_llm_generation: bool = False,
    llm_cpu_fast_mode: bool = LLM_CPU_FAST_MODE,
    llm_bypass_in_llm_mode: bool = LLM_BYPASS_IN_LLM_MODE,
    llm_enforce_latency_budget: bool = LLM_ENFORCE_LATENCY_BUDGET,
) -> Tuple[str, Dict[str, str]]:
    parts = split_compound_query(query)
    if not parts:
        return "Please enter your admissions question first.", {
            "source": "empty_query",
            "confidence": "0.0",
            "matched_question": "",
        }

    # Single question path keeps existing behavior.
    if len(parts) == 1:
        return _get_single_answer(
            parts[0],
            index,
            entries,
            questions,
            use_llm_generation=use_llm_generation,
            llm_cpu_fast_mode=llm_cpu_fast_mode,
            llm_bypass_in_llm_mode=llm_bypass_in_llm_mode,
            llm_enforce_latency_budget=llm_enforce_latency_budget,
        )

    # Compound path: answer each part independently, then compose.
    combined_blocks = []
    matched = []
    confidence_items = []
    llm_used = False
    seen_answer_blocks = set()
    for i, part in enumerate(parts, start=1):
        ans, meta = _get_single_answer(
            part,
            index,
            entries,
            questions,
            use_llm_generation=use_llm_generation,
            llm_cpu_fast_mode=llm_cpu_fast_mode,
            llm_bypass_in_llm_mode=llm_bypass_in_llm_mode,
            llm_enforce_latency_budget=llm_enforce_latency_budget,
        )
        ans_key = normalize_text(ans)
        if ans_key in seen_answer_blocks:
            continue
        seen_answer_blocks.add(ans_key)
        combined_blocks.append(f"{len(combined_blocks) + 1}. {ans}")
        m = meta.get("matched_question", "")
        if m:
            matched.append(m)
        c = meta.get("confidence", "")
        if c:
            confidence_items.append(c)
        if meta.get("source") == "llm_grounded":
            llm_used = True

    return "\n\n".join(combined_blocks), {
        "source": "compound_llm" if llm_used else "compound_retrieval",
        "confidence": " | ".join(confidence_items[:3]),
        "matched_question": " | ".join(matched[:2]),
    }


def source_note(meta: Dict[str, str]) -> str:
    src = meta.get("source", "")
    conf = meta.get("confidence", "")
    if src == "llm_grounded":
        return f"Source: LLM grounded on retrieved FAQ context ({conf})."
    if src == "faq_fallback_after_llm_error":
        return f"Source: Direct FAQ fallback because LLM failed ({conf})."
    if src == "faq_fallback_after_guardrail":
        return f"Source: Guardrailed fallback to top grounded FAQ ({conf})."
    if src == "fast_retrieval":
        return f"Source: Fast retrieval mode (LLM skipped) ({conf})."
    if src == "retrieval_only":
        return f"Source: Retrieval-only mode (LLM disabled) ({conf})."
    if src == "intent_override":
        return "Source: Intent-aware exact FAQ routing."
    if src == "compound_retrieval":
        return f"Source: Compound query split + retrieval mode ({conf})."
    if src == "compound_llm":
        return f"Source: Compound query split + LLM mode ({conf})."
    if src == "small_talk":
        return "Source: Conversational assistant behavior."
    if src == "empty_query":
        return "Source: Input validation."
    return "Source: Not found confidently in official FAQ."


def render_chat_bubble(answer: str) -> None:
    # The bubble is rendered as HTML, so convert markdown links to <a> tags first.
    link_pattern = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
    parts = []
    cursor = 0
    for match in link_pattern.finditer(answer):
        parts.append(html.escape(answer[cursor:match.start()]))
        label = html.escape(match.group(1))
        url = html.escape(match.group(2), quote=True)
        parts.append(f'<a href="{url}" target="_blank" rel="noopener noreferrer">{label}</a>')
        cursor = match.end()
    parts.append(html.escape(answer[cursor:]))
    safe_answer_html = "".join(parts)

    st.markdown(
        f"""
        <div style="
            border: 1px solid #d7e3f4;
            border-radius: 12px;
            padding: 14px 16px;
            background: #f7fbff;
            color: #1f2d3d;
            line-height: 1.5;
            white-space: pre-wrap;
            margin-top: 8px;
            margin-bottom: 10px;
        ">
            {safe_answer_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="NUST Admissions Assistant - LLM", page_icon="🎓", layout="centered")

    st.title("NUST Admissions Assistant (LLM-First)")
    st.write("This version uses a local LLM as primary answer engine, grounded on your NUST FAQ JSON.")

    try:
        entries, questions, _, faq_path = load_faq_data()
        index, _ = build_index(questions)
        model_path = _resolve_model_path()
    except Exception as exc:
        st.error(f"Initialization failed: {exc}")
        st.stop()

    st.caption(f"FAQ: {faq_path}")
    st.caption(f"LLM: {model_path}")

    speed_mode = st.radio(
        "Response mode",
        options=["Fast (recommended)", "Accurate (LLM, slower)"],
        horizontal=True,
        help="Fast mode skips LLM generation and returns grounded FAQ answers quickly. Accurate mode uses LLM and may be much slower on CPU.",
    )
    use_llm_generation = speed_mode == "Accurate (LLM, slower)"

    llm_profile = st.sidebar.radio(
        "LLM Profile",
        options=["Accuracy", "Balanced", "Speed"],
        index=0,
        help="Applies when Response mode is Accurate (LLM, slower).",
    )
    if llm_profile == "Accuracy":
        llm_cpu_fast_mode = False
        llm_bypass_in_llm_mode = False
        llm_enforce_latency_budget = False
    elif llm_profile == "Balanced":
        llm_cpu_fast_mode = False
        llm_bypass_in_llm_mode = True
        llm_enforce_latency_budget = False
    else:
        llm_cpu_fast_mode = True
        llm_bypass_in_llm_mode = True
        llm_enforce_latency_budget = True

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.form("ask_form", clear_on_submit=False):
        query = st.text_input("Ask your admissions question", placeholder="Example: Whaat is fee at NuST?")
        submitted = st.form_submit_button("Ask")

    if submitted:
        answer, meta = get_answer(
            query,
            index,
            entries,
            questions,
            use_llm_generation=use_llm_generation,
            llm_cpu_fast_mode=llm_cpu_fast_mode,
            llm_bypass_in_llm_mode=llm_bypass_in_llm_mode,
            llm_enforce_latency_budget=llm_enforce_latency_budget,
        )
        st.session_state.history.append({"query": query, "answer": answer, "meta": meta})

    if st.session_state.history:
        st.subheader("Conversation")
        for item in reversed(st.session_state.history):
            st.markdown(f"**You:** {item['query']}")
            render_chat_bubble(item["answer"])
            st.caption(source_note(item["meta"]))
            matched = item["meta"].get("matched_question", "")
            if matched:
                st.caption(f"Matched FAQ: {matched}")
            st.markdown("---")


if __name__ == "__main__":
    main()
