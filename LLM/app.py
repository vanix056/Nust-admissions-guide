import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
LLM_TOP_K = 4
LLM_MAX_TOKENS = 140
LLM_CTX_SIZE = 2048

UNKNOWN_MESSAGE = (
    "I could not find a reliable answer in the official NUST FAQ knowledge base. "
    "Please try rephrasing your question.\n\n"
    "For further guidance, contact NUST Undergraduate Admissions: "
    "https://nust.edu.pk/admissions/"
)


# --------------------------
# Utilities
# --------------------------
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


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
    text = re.sub(r"Please visit:\s*", "Please visit: ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(Please visit:\s*https?://[^\s]+)(?:\s+\1)+", r"\1", text, flags=re.IGNORECASE)

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


def _is_fee_intent(query_norm: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", query_norm))
    return "fee" in tokens or ("fee" in query_norm and "structure" in query_norm)


def _general_fee_faq_index(questions: List[str], answers: List[str]) -> int:
    for i, answer in enumerate(answers):
        if "undergraduate-financial-matters" in answer.lower():
            return i

    for i, question in enumerate(questions):
        q = normalize_text(question)
        if "fee structure" in q and ("different" in q or "ug" in q or "undergraduate" in q):
            return i

    return -1


def _resolve_faq_path() -> Path:
    candidates = [
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
def response_cache() -> Dict[str, Tuple[str, Dict[str, str]]]:
    """Simple in-memory cache for repeated normalized queries."""
    return {}


@st.cache_resource(show_spinner=False)
def load_faq_data() -> Tuple[List[Dict[str, str]], List[str], List[str], Path]:
    faq_path = _resolve_faq_path()
    with faq_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("FAQ JSON must be a list of objects.")

    entries: List[Dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if q and a:
            entries.append({"question": q, "answer": a})

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
    cpu_threads = max(2, min(8, (os.cpu_count() or 4)))
    return Llama(
        model_path=str(model_path),
        n_ctx=LLM_CTX_SIZE,
        n_threads=cpu_threads,
        n_batch=256,
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
    if _is_fee_intent(query_for_match):
        fee_idx = _general_fee_faq_index(questions, answers)
        if fee_idx >= 0 and not any(k in query_for_match for k in ["bshnd", "mbbs", "nshs"]):
            ordered = [fee_idx] + [i for i in ordered if i != fee_idx]

    return ordered, best_sem, float(best_fuzzy)


def build_grounded_prompt(query: str, candidate_indices: List[int], entries: List[Dict[str, str]]) -> str:
    context_parts = []
    for rank, idx in enumerate(candidate_indices[:LLM_TOP_K], start=1):
        item = entries[idx]
        context_parts.append(f"[{rank}] Q: {item['question']}\n[{rank}] A: {item['answer']}")

    context = "\n\n".join(context_parts)
    return (
        "You are a careful NUST admissions assistant. "
        "Answer ONLY using the provided FAQ context. "
        "If the answer is not in context, clearly say you do not know and ask user to contact admissions.\n\n"
        f"FAQ Context:\n{context}\n\n"
        f"Student Question: {query}\n\n"
        "Instructions:\n"
        "1) Prefer item [1] by default. Only move to other items if [1] is clearly irrelevant.\n"
        "2) Prefer the most general relevant answer unless user explicitly asks for a specific program.\n"
        "3) Keep answer concise and factual.\n"
        "4) Include only links exactly present in the context; never invent links.\n"
        "5) Do not output tags like <|assistant|>, 'Solution:', or 'Final Answer:'.\n\n"
        "Final Answer:"
    )


def _strip_generation_artifacts(text: str) -> str:
    cleaned = text
    cleaned = cleaned.replace("<|assistant|>", "")
    cleaned = cleaned.replace("Solution:", "")
    cleaned = cleaned.replace("Final Answer:", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_links(text: str) -> List[str]:
    return re.findall(r"https?://[^\s)]+", text)


def llm_grounded_answer(query: str, prompt: str, allowed_links: List[str]) -> str:
    model = load_llm()
    out = model(
        prompt,
        max_tokens=LLM_MAX_TOKENS,
        temperature=0,
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


def get_answer(
    query: str,
    index: faiss.IndexFlatIP,
    entries: List[Dict[str, str]],
    questions: List[str],
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

    vocab = build_question_vocab(questions)
    query_for_match = normalize_query_for_matching(query, vocab)
    cached = response_cache().get(query_for_match)
    if cached:
        return cached

    candidate_indices, best_sem, best_fuzzy = retrieve_candidate_indices(query_for_match, index, questions, [e["answer"] for e in entries])

    if best_sem < MIN_SEMANTIC_CONF and best_fuzzy < MIN_FUZZY_CONF:
        result = (UNKNOWN_MESSAGE, {
            "source": "unknown",
            "confidence": f"semantic={best_sem:.3f}, fuzzy={best_fuzzy:.1f}",
            "matched_question": "",
        })
        response_cache()[query_for_match] = result
        return result

    top_idx = candidate_indices[0] if candidate_indices else 0

    # Fast-path: for very confident retrieval, skip LLM generation.
    if best_sem >= FAST_RETURN_SEMANTIC or best_fuzzy >= FAST_RETURN_FUZZY:
        result = (
            format_answer_text(entries[top_idx]["answer"]),
            {
                "source": "fast_retrieval",
                "confidence": f"semantic={best_sem:.3f}, fuzzy={best_fuzzy:.1f}",
                "matched_question": entries[top_idx]["question"],
            },
        )
        response_cache()[query_for_match] = result
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
        answer = format_answer_text(entries[top_idx]["answer"])
        result = (answer, {
            "source": "faq_fallback_after_llm_error",
            "confidence": f"semantic={best_sem:.3f}, fuzzy={best_fuzzy:.1f}",
            "matched_question": entries[top_idx]["question"],
            "error": str(exc),
        })
        response_cache()[query_for_match] = result
        return result

    # If generation is empty after guardrails, use top grounded FAQ.
    if not answer:
        result = (format_answer_text(entries[top_idx]["answer"]), {
            "source": "faq_fallback_after_guardrail",
            "confidence": f"semantic={best_sem:.3f}, fuzzy={best_fuzzy:.1f}",
            "matched_question": entries[top_idx]["question"],
        })
        response_cache()[query_for_match] = result
        return result

    # Additional safety: generic fee queries should not drift to specific programs.
    qn = normalize_text(query_for_match)
    if _is_fee_intent(qn) and not any(k in qn for k in ["bshnd", "mbbs", "nshs"]):
        if any(k in normalize_text(answer) for k in ["bshnd", "mbbs", "nshs"]):
            result = (format_answer_text(entries[top_idx]["answer"]), {
                "source": "faq_fallback_after_guardrail",
                "confidence": f"semantic={best_sem:.3f}, fuzzy={best_fuzzy:.1f}",
                "matched_question": entries[top_idx]["question"],
            })
            response_cache()[query_for_match] = result
            return result

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    result = (answer, {
        "source": "llm_grounded",
        "confidence": f"semantic={best_sem:.3f}, fuzzy={best_fuzzy:.1f}, llm_ms={elapsed_ms}",
        "matched_question": entries[top_idx]["question"],
    })
    response_cache()[query_for_match] = result
    return result


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
    if src == "small_talk":
        return "Source: Conversational assistant behavior."
    if src == "empty_query":
        return "Source: Input validation."
    return "Source: Not found confidently in official FAQ."


def render_chat_bubble(answer: str) -> None:
    st.markdown(
        f"""
        <div style="
            border: 1px solid #d7e3f4;
            border-radius: 12px;
            padding: 14px 16px;
            background: #f7fbff;
            color: #1f2d3d;
            line-height: 1.5;
            margin-top: 8px;
            margin-bottom: 10px;
        ">
            {answer}
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

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.form("ask_form", clear_on_submit=False):
        query = st.text_input("Ask your admissions question", placeholder="Example: Whaat is fee at NuST?")
        submitted = st.form_submit_button("Ask")

    if submitted:
        answer, meta = get_answer(query, index, entries, questions)
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
